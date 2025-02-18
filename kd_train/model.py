import contextlib
import torch
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap

from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    WhisperFeatureExtractor,
    WhisperModel,
    LlamaForCausalLM,
    LlamaTokenizer
)
import librosa
from models.beats.BEATs import BEATsConfig, BEATs
from models.Qformer import BertConfig, BertLMHeadModel
from transformers import BitsAndBytesConfig

import os

from torch.distributed.pipelining import PipelineStage

# --- Stage0: embed_tokens와 디코더 레이어 0~15를 처리하는 모듈 ---
class LlamaStage0(nn.Module):
    def __init__(self, full_llama_model, split_index=16):
        super().__init__()
        self.embed_tokens = full_llama_model.model.embed_tokens
        # rotary_emb는 반드시 보존
        self.rotary_emb = full_llama_model.model.rotary_emb
        self.layers = nn.ModuleList([full_llama_model.model.layers[i] for i in range(split_index)])
    def forward(self, inputs_embeds, attention_mask):
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        return hidden_states

class LlamaStage1(nn.Module):
    def __init__(self, full_llama_model, split_index=16):
        super().__init__()
        # rotary_emb를 보존
        self.rotary_emb = full_llama_model.model.rotary_emb
        self.layers = nn.ModuleList([full_llama_model.model.layers[i] for i in range(split_index, len(full_llama_model.model.layers))])
        self.norm = full_llama_model.model.norm
        self.lm_head = full_llama_model.lm_head
    def forward(self, hidden_states, attention_mask):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

class LlamaPipelineWrapper(nn.Module):
    def __init__(self, stage0: PipelineStage, stage1: PipelineStage, max_seq_length=512):
        super().__init__()
        self.stage0 = stage0
        self.stage1 = stage1
        # 기본 position_ids를 max_seq_length 길이의 텐서로 생성하여 buffer로 등록
        # shape: (1, max_seq_length)
        self.register_buffer("default_position_ids", torch.arange(max_seq_length).unsqueeze(0))
    
    def forward(self, inputs_embeds, attention_mask, position_ids=None):
        batch_size, seq_len, _ = inputs_embeds.shape
        # 만약 position_ids가 None이라면, default_position_ids를 사용 (batch_size, seq_len)로 확장
        if position_ids is None:
            position_ids = self.default_position_ids[:, :seq_len].expand(batch_size, -1)
        # 여기서 만약 Stage0 또는 Stage1에서 position_ids가 필요하다면, 
        # 해당 인자를 추가로 전달해야 합니다.
        # 예를 들어, 아래와 같이 stage0.forward와 stage1.forward에 추가 인자를 전달할 수 있도록 수정할 수 있습니다.
        print(">> LlamaPipelineWrapper.forward: inputs_embeds.shape =", inputs_embeds.shape)
        print(">> LlamaPipelineWrapper.forward: attention_mask.shape =", attention_mask.shape)
        hidden_states = self.stage0(inputs_embeds, attention_mask)
        print(">> After Stage0: hidden_states.shape =", hidden_states.shape)
        hidden_states = hidden_states.to("cuda:1")
        logits = self.stage1(hidden_states, attention_mask)
        print(">> After Stage1: logits.shape =", logits.shape)
        return logits


class SALMONN(nn.Module):
    def __init__(
        self,
        ckpt,
        whisper_path,
        beats_path,
        vicuna_path,
        speech_qformer_token_num=1,
        speech_qformer_layer=2,
        lora=True,
        lora_alpha=32,
        lora_rank=8,
        lora_dropout=0.1,
        second_per_frame=0.333333,
        second_stride=0.333333,
        low_resource=True,
        device=None,
        quantization="8bit",
    ):

        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        # YAML 
        self.config = self.from_config(config_path="configs/train_stage1_kd.yaml")

        self.freeze_whisper = self.config["freeze_whisper"]
        self.freeze_beats = self.config["freeze_beats"]

        self.use_speech_Qformer = self.config["use_speech_Qformer"]
        self.num_speech_query_token = self.config["num_speech_query_token"]
        self.freeze_speech_QFormer = self.config["freeze_speech_QFormer"]
        self.window_level_Qformer = self.config["window_level_Qformer"]
        self.second_per_window = self.config["second_per_window"]
        self.second_stride = self.config["second_stride"]

        self.speech_llama_proj_model = self.config["speech_llama_proj_model"]
        self.freeze_speech_llama_proj = self.config["freeze_speech_llama_proj"]

        self.lora = self.config["lora"]
        self.lora_rank = self.config["lora_rank"]
        self.lora_alpha = self.config["lora_alpha"]
        self.lora_dropout = self.config["lora_dropout"]

        self.multi_prompt = self.config["multi_prompt"]
        self.max_txt_len = self.config["max_txt_len"]
        self.end_sym = self.config["end_sym"]
        self.low_resource = self.config["low_resource"]
        self.token = self.config["token"]
        
        # feature_extractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        
        # BitsAndBytes Config (8bit 또는 4bit)
        if quantization == "8bit":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            quant_config = None

        # Whisper
        self.speech_encoder = WhisperModel.from_pretrained(whisper_path,quantization_config=quant_config).encoder
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)
        for param in self.speech_encoder.parameters():
            param.requires_grad = False
            
        # BEATs
        self.beats_ckpt = beats_path
        beats_checkpoint = torch.load(self.beats_ckpt, map_location='cpu')
        beats_cfg = BEATsConfig(beats_checkpoint['cfg'])
        beats = BEATs(beats_cfg)
        beats.load_state_dict(beats_checkpoint['model'])
        self.beats = beats
        self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
        for name, param in self.beats.named_parameters():
            param.requires_grad = False
        self.beats.eval()

        # init speech Qformer
        self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
            speech_qformer_token_num,
            self.speech_encoder.config.d_model + self.beats.cfg.encoder_embed_dim,
            speech_qformer_layer,
        )
        self.second_per_frame = second_per_frame
        self.second_stride = second_stride
            
        self.llama_model = LlamaForCausalLM.from_pretrained(
                vicuna_path,
                torch_dtype=torch.float16,
                quantization_config=quant_config
            ).to(self.device)

        self.llama_model.gradient_checkpointing_enable()
        # self.llama_model = torch.compile(self.llama_model)

        # lora
        self.lora = lora
        if self.lora:
            target_modules = None
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=True, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
                target_modules=target_modules,
            )
            # self.llama_model = get_peft_model(self.llama_model, self.peft_config)
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)
            self.llama_model = self.llama_model.merge_and_unload()  # ✅ 병합을 방지하여 VRAM 절약

        # tokenizer
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(vicuna_path, use_fast=False)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
        self.llama_tokenizer.padding_side = "right"

        # proj
        self.speech_llama_proj = nn.Linear(
            self.speech_Qformer.config.hidden_size, self.llama_model.config.hidden_size)

        # load ckpt
        ckpt_dict = torch.load(ckpt)['model']
        self.load_state_dict(ckpt_dict, strict=False)

        self.to(self.device)
        self.cleanup()
        
        print(f'llama_layer is {self.llama_model.model.layers}')
        #################### 
        import copy

        # --- LLaMA 모델을 파이프라인으로 분할 ---
        # deep copy를 이용해 각 스테이지에 독립적인 복사본을 만듭니다.
        # Stage0: embed_tokens와 layers[0:16]를 사용. norm, lm_head는 필요 없으므로 그대로 두지 않습니다.
        full_model_stage0 = copy.deepcopy(self.llama_model)
        # ※ 여기서 rotary_emb는 삭제하지 않고 그대로 보존해야 합니다.
        full_model_stage0.model.norm = None
        full_model_stage0.lm_head = None
        
        # 만약 rotary_emb.inv_freq가 None이면 재초기화 (예: config를 사용)
        if full_model_stage0.model.rotary_emb.inv_freq is None:
            orig_inv_freq = self.llama_model.model.rotary_emb.inv_freq
            if orig_inv_freq is not None:
                full_model_stage0.model.rotary_emb.register_buffer("inv_freq", orig_inv_freq.clone())
        stage0_module = LlamaStage0(full_model_stage0, split_index=16)
        example_input = (torch.randn(1, 128, self.llama_model.config.hidden_size, device="meta"),
                         torch.ones(1, 128, device="meta"))
        self.stage0 = PipelineStage(stage0_module, stage_index=0, num_stages=2, device="cuda:0", input_args=example_input)
        
        # Stage1: layers[16:] + norm + lm_head; embed_tokens는 사용하지 않음
        full_model_stage1 = copy.deepcopy(self.llama_model)
        # embed_tokens는 사용하지 않으므로 제거해도 되지만, rotary_emb를 보존해야 합니다.
        full_model_stage1.model.layers = nn.ModuleList(list(full_model_stage1.model.layers[16:]))
        stage1_module = LlamaStage1(full_model_stage1, split_index=0)  # split_index 인자는 사용하지 않음
        self.stage1 = PipelineStage(stage1_module, stage_index=1, num_stages=2, device="cuda:1", input_args=example_input)
        
        # 최종 파이프라인 래퍼
        self.pipe_llama = LlamaPipelineWrapper(self.stage0, self.stage1)


    def generate(
        self,
        wav_path,
        prompt,
        prompt_pattern="USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:",
        device='cuda:0',
        max_length=150,
        num_beams=4,
        do_sample=True,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=1.0,
    ):
        # read wav
        wav, sr = sf.read(wav_path)
        if len(wav.shape) == 2:
            wav = wav[:, 0]
        if len(wav) > 30 * sr:
            wav = wav[: 30 * sr]
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="fft")
        
        # whisper
        spectrogram = self.feature_extractor(wav, return_tensors="pt", sampling_rate=16000).input_features.to(device) # [1, 80, 3000]
        speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state
       
        # beats
        raw_wav = torch.from_numpy(wav).to(device).unsqueeze(0)
        audio_padding_mask = torch.zeros(raw_wav.shape, device=device).bool()
        audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)

        # auditory embeds
        speech_embeds = self.ln_speech(speech_embeds)
        audio_embeds = self.ln_audio(audio_embeds)
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
        speech_embeds = torch.cat([speech_embeds, audio_embeds], dim=-1)

        # split frames
        B, T, C = speech_embeds.shape
        kernel = round(T * self.second_per_frame / 30.0)
        stride = round(T * self.second_stride / 30.0)
        kernel = (1, kernel)
        stride = (1, stride)
        speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
        speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
        _, _, L = speech_embeds_overlap.shape
        speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
        speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
        speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

        # Qformer
        query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
        query_output = self.speech_Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)
        speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

        # USER: <Speech>speech_embeds<Speech> prompt\nASSISTANT:
        embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens
        prompt_left, prompts_right = prompt_pattern.format(prompt).split('<SpeechHere>')
        prompt_left_ids = self.llama_tokenizer(
            prompt_left,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_left_embeds = embed_tokens(prompt_left_ids)
        prompt_right_ids = self.llama_tokenizer(
            prompts_right,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_right_embeds = embed_tokens(prompt_right_ids)

        bos_embeds = self.llama_model.model.embed_tokens(
            torch.ones(
                [1, 1],
                dtype=torch.long,
                device=device,
            ) * self.llama_tokenizer.bos_token_id
        ) if not self.lora else self.llama_model.model.model.embed_tokens(
            torch.ones(
                [1, 1],
                dtype=torch.long,
                device=device,
            ) * self.llama_tokenizer.bos_token_id
        )

        embeds = torch.cat([bos_embeds, prompt_left_embeds, speech_embeds, prompt_right_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)

        def check_dtype(tensor, name):
            if isinstance(tensor, torch.Tensor):
                print(f"{name} dtype: {tensor.dtype}, shape: {tensor.shape}")
            elif isinstance(tensor, list) and len(tensor) > 0 and isinstance(tensor[0], torch.Tensor):
                print(f"{name} (list of tensors) first dtype: {tensor[0].dtype}, shape: {tensor[0].shape}")
            else:
                print(f"{name} is not a tensor")

        expected_dtype = next(self.llama_model.parameters()).dtype
        # print(f"🔥 LLaMA Model expected dtype: {expected_dtype}")
        speech_embeds = speech_embeds.to(dtype=expected_dtype)
        audio_embeds = audio_embeds.to(dtype=expected_dtype)
        prompt_left_embeds = prompt_left_embeds.to(dtype=expected_dtype)
        prompt_right_embeds = prompt_right_embeds.to(dtype=expected_dtype)
        bos_embeds = bos_embeds.to(dtype=expected_dtype)
        embeds = embeds.to(dtype=expected_dtype)
        atts = atts.to(dtype=torch.long)  # Attention Mask는 int64 유지

        # generate
        output = self.llama_model.generate(
            inputs_embeds=embeds,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            attention_mask=atts,
            bos_token_id=self.llama_tokenizer.bos_token_id,
            eos_token_id=self.llama_tokenizer.eos_token_id,
            pad_token_id=self.llama_tokenizer.pad_token_id
        )
        
        output_text = self.llama_tokenizer.batch_decode(output, add_special_tokens=False, skip_special_tokens=True)
        print(f"Generated Output (Before Returning): {output_text}")

        return output_text
                
        
    def init_speech_Qformer(self, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig()
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    
    # 처리된 samples가 들어옴
    def forward(self, samples, verboase=False):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):

            prompt = samples["prompt"]
            print(f'Teacher Model Prompt: {prompt}')
            spectrogram = samples["spectrogram"].to(self.device)  # (B, ...)
            raw_wav = samples.get("raw_wav", None).to(self.device)
            audio_padding_mask = samples.get("padding_mask", None).to(self.device)
            speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav, audio_padding_mask)

            # if samples['prompt_dict'] and prompt:
            #     speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)
            if isinstance(samples['prompt'], list):
                speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, samples["prompt"], multi_prompt=True)

        # # LLaMA 입력 준비
        # batch_size = speech_embeds.shape[0]
        # bos = torch.ones((batch_size, 1), dtype=torch.long, device=speech_embeds.device) * self.llama_tokenizer.bos_token_id
        # bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        
        # inputs_embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        # attention_mask = torch.cat([torch.ones_like(bos), speech_atts], dim=1)
        # targets = None
        
            # prepare inputs for LLM
            text = [t + self.end_sym for t in samples["text"]]
            to_regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(self.device)
            embed_tokens = self.get_embed_tokens()

            to_regress_embeds = embed_tokens(to_regress_tokens.input_ids) 
            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )
            empty_targets = (
                torch.ones(
                    [speech_atts.shape[0], speech_atts.shape[1] + 1],
                    dtype=torch.long,
                    device=self.device
                ).fill_(-100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            batch_size = speech_embeds.shape[0]
            bos = torch.ones(
                [batch_size, 1],
                dtype=to_regress_tokens.input_ids.dtype,
                device=self.device,
            ) * self.llama_tokenizer.bos_token_id

            bos_embeds = self.get_embed_tokens()(bos)
            atts_bos = speech_atts[:, :1]

            inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1).to(self.device)
            attention_mask = torch.cat([atts_bos, speech_atts, to_regress_tokens.attention_mask], dim=1).to(self.device)

            
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=None  # targets가 None이면 loss 계산하지 않음
            )
            self.cleanup()
            print(f'self.llama_model is {self.llama_model}')
            
            if  self.pipe_llama is not None:
                logits = self.pipe_llama(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,  
                )
                self.cleanup()
                return {"logits": logits}
            else:
                outputs = self.llama_model(     
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=None
                )
                self.cleanup()
                return {"logits": outputs.logits}

    def _encode_auditory_feature(self, speech_embeds, audio_embeds=None):
        with self.maybe_autocast():
            if self.use_speech_Qformer:
                speech_embeds = self.ln_speech(speech_embeds)
                if audio_embeds is not None:
                    audio_embeds = self.ln_audio(audio_embeds)
                    if audio_embeds.size(1) < speech_embeds.size(1):
                        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
                    elif audio_embeds.size(1) > speech_embeds.size(1):
                        speech_embeds = F.pad(speech_embeds, (0, 0, 0, audio_embeds.size(1) - speech_embeds.size(1)))
                    speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1)
                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

                if self.window_level_Qformer:
                    B, T, C = speech_embeds.shape
                    kernel = round(1500 * self.second_per_window / 30.0)
                    stride = round(1500 * self.second_stride / 30.0)
                    kernel = (1, kernel)
                    stride = (1, stride)
                    speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
                    speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
                    _, _, L = speech_embeds_overlap.shape
                    speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
                    speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
                    speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
                    speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

                query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
                query_output = self.speech_Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=speech_embeds,
                    encoder_attention_mask=speech_atts,
                    return_dict=True,
                )
                speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)

                if self.window_level_Qformer:
                    speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()

                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
            else:
                raise NotImplementedError

        return speech_embeds, speech_atts

    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        with self.maybe_autocast():
            speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state

            if self.beats_ckpt and raw_wav is not None:
                audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)
            else:
                audio_embeds = None
                        
        return self._encode_auditory_feature(speech_embeds, audio_embeds=audio_embeds)


    def prompt_wrap(self, embeds, atts, prompt, multi_prompt=False):
        if prompt:
            if multi_prompt:
                p_before = []
                p_after = []
                for i, p in enumerate(prompt):
                    
                    b, a = p.split("<SpeechHere>")
                    p_before.append(b)
                    p_after.append(a)
                
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids)

                # speech_embeds wrapped with prompts_embeds are padded to the same length here
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", padding="longest", add_special_tokens=False
                ).to(embeds.device)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            else:
                batch_size = embeds.shape[0]
                p_before, p_after = prompt.split("<SpeechHere>")

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            return wrapped_embeds, wrapped_atts
        else:
            return embeds, atts
        
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast(device_type='cuda', dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def from_config(cls, config_path="configs/train_stage1_kd.yaml"):
        """
        YAML 파일에서 모델 설정을 로드하여 SALMONN 클래스의 인스턴스를 생성
        """
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config['model']

    def get_embed_tokens(self):
        """
        임베딩 레이어(embed_tokens)를 올바른 경로로 접근하여, 지정된 self.device로 이동시켜 반환합니다.
        """
        # print(f'llama_model is {self.llama_model}')
        embed_tokens = self.llama_model.model.embed_tokens
        # if self.lora:
        #     # PEFT(Lora) 적용된 경우
        #     if hasattr(self.llama_model, "base_model"):
        #         embed_tokens = self.llama_model.base_model.model.model.embed_tokens
        #     else:
        #         embed_tokens = self.llama_model.model.model.embed_tokens
        # else:
        #     # Lora 미적용: DDP 여부 확인
        #     if hasattr(self.llama_model, "base_model"):
        #         embed_tokens = self.llama_model.base_model.model.model.embed_tokens
        #     else:
        #         embed_tokens = self.llama_model.model.model.embed_tokens
    
        for param in embed_tokens.parameters():
            param.data = param.data.to(self.device)

        return embed_tokens.to(self.device)

    def cleanup(self):
        import gc
        gc.collect()
        torch.cuda.empty_cache()