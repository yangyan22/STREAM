import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
import pdb
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class STREAM(pl.LightningModule):
    """
    STREAM model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["query", "value"],
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')
        print('Loading tiny llama')
        self.llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model)
        self.llama_tokenizer.pad_token_id = 0
        self.llama_model = AutoModelForCausalLM.from_pretrained(args.llama_model, torch_dtype=torch.float16, trust_remote_code=True)
        self.embed_tokens = self.llama_model.get_input_embeddings()
        print('Loading tinyllama done')
        
        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r, lora_alpha=args.llm_alpha, lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LLAMA LoRA Done')         
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done')

        
        self.bert_tokenizer = AutoTokenizer.from_pretrained(args.cxr_bert_path, trust_remote_code=True)
        self.text_encoder = AutoModel.from_pretrained(args.cxr_bert_path, trust_remote_code=True)
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
        print(f'Loading Frozen text encoder:{args.cxr_bert_path} -- Done')

        self.text_crossattention_block = nn.MultiheadAttention(768, 12, dropout=0.1, add_bias_kv=True,
                                                                       add_zero_attn=True)
        self.text_layernorm_768_1 = nn.LayerNorm(768)
        self.text_llama_proj_1 = nn.Linear(768, 768)
        self.text_layernorm_768_2 = nn.LayerNorm(768)
        self.text_llama_proj_2 = nn.Linear(768, 2048)
        self.text_llama_proj_3 = nn.Linear(2048, 4096)
        self.text_layernorm_4096_1 = nn.LayerNorm(4096)
        self.text_image= nn.Linear(2048, 768)
        self.iit_proj = nn.Linear(4096,2048)
        self.iit_layernorm = nn.LayerNorm(2048)

        self.new_embeds_proj = nn.Linear(4096,2048)
        self.new_embeds_layernorm = nn.LayerNorm(2048)

        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
      
        self.prompt = 'Please generate a detailed chest X-ray report, including descriptions of any abnormalities or significant findings, along with the view and temporal progression details.'
        self.val_step_outputs = []
        self.val_step_ref = []
        self.val_step_ids = []
        self.test_step_outputs = []
        self.val_score = 0.0

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

        self.fusion = nn.MultiheadAttention(self.visual_encoder.num_features, 8, batch_first=True)
        self.fusion1 = nn.MultiheadAttention(self.visual_encoder.num_features, 8, batch_first=True)
        self.fusion2 = nn.MultiheadAttention(self.visual_encoder.num_features, 8, batch_first=True)

        
    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores
    
    def encode_text(self, text, query_feature):
        tokens = self.bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=80,
        )
        input_ids = tokens["input_ids"].to(query_feature.device)
        token_type_ids = tokens["token_type_ids"].to(query_feature.device)
        attention_mask = tokens["attention_mask"].to(query_feature.device)
        output = self.text_encoder(input_ids, attention_mask, token_type_ids,
                        return_dict=True, mode="text")
        outputs = output.last_hidden_state
        #print(query_feature.transpose(0, 1).shape)
        #print(outputs.transpose(0, 1).shape)
        inputs_llama = self.text_crossattention_block(query_feature.transpose(0, 1),
                                                         outputs.transpose(0, 1),
                                                         outputs.transpose(0, 1))[0].transpose(0, 1)  
        inputs_llama = self.text_layernorm_768_1(inputs_llama)
        inputs_llama1 = self.text_llama_proj_1(inputs_llama)
        inputs_llama1 = self.text_layernorm_768_2(inputs_llama1) + inputs_llama
        inputs_llama1 = self.text_llama_proj_2(inputs_llama1)
        inputs_llama1 = F.gelu(inputs_llama1)
        inputs_llama1 = self.text_llama_proj_3(inputs_llama1)
        inputs_llama1 = self.text_layernorm_4096_1(inputs_llama1)
        atts_llama = torch.ones(inputs_llama1.size()[:-1], dtype=torch.long).to(query_feature.device)
        return inputs_llama1, atts_llama

    def encode_img(self, images, bs, temp_images1, temp_images2):
        device = images.device
        # print(images.shape) # torch.Size([4, 3, 3, 224, 224])
         
        image_embed = self.visual_encoder(images.view(-1, 3, 224, 224))['last_hidden_state'].to(device)
        temp_image_embed1, temp_image_embed2 = None, None
        temp_image_embed1 = self.visual_encoder(temp_images1.view(-1, 3, 224, 224))['last_hidden_state'].to(device)
        temp_image_embed2 = self.visual_encoder(temp_images2.view(-1, 3, 224, 224))['last_hidden_state'].to(device)

        # print(image_embed.shape)  # torch.Size([12, 49, 1024])
        # print(temp_image_embed1.shape)  # torch.Size([12, 49, 1024])
        # print(temp_image_embed2.shape)  # torch.Size([12, 49, 1024])
        image_embed = image_embed.view( bs, 3, 49, 1024)
        temp_image_embed1 = temp_image_embed1.view(bs, 3, 49, 1024)
        temp_image_embed2 = temp_image_embed2.view(bs, 3, 49, 1024)
        # print(image_embed.shape)  # torch.Size([bs, 49, 1024])
        # print(temp_image_embed1.shape)  # torch.Size([bs, 49, 1024])
        # print(temp_image_embed2.shape)  # torch.Size([bs, 49, 1024])

        new_image_embeds = self.process_image_sets(image_embed, temp_image_embed1, temp_image_embed2)
        # print(new_image_embeds.shape)   # torch.Size([4, 147, 1024])
        # print("END")
        inputs_llama = self.llama_proj(new_image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embed.device)
        # print(atts_llama)
        # print(atts_llama.shape)  # 4 147
        return inputs_llama, atts_llama
        
    def fuse_and_sum(self, embeds, start_idx, count):
        # new_embed = embeds[start_idx]
        new_embed = None
        for i in range(start_idx, start_idx + count):
            for j in range(i + 1, start_idx + count): 
                # print(embeds[:, i, :, :].shape)  # torch.Size([4, 49, 1024])
                fusion_embed = self.fusion(embeds[:, i, :, :] , embeds[:, j, :, :], embeds[:, j, :, :])[0]
                reverse_fusion_embed = self.fusion(embeds[:, j, :, :], embeds[:, i, :, :], embeds[:, i, :, :])[0]
                if new_embed == None:
                    new_embed = fusion_embed + reverse_fusion_embed
                else:
                    new_embed = new_embed + fusion_embed + reverse_fusion_embed
      
        for i in range(start_idx, start_idx + count):
            if new_embed is None :
                new_embed = embeds[:, i, :, :]
            else:
                new_embed += embeds[:, i, :, :]
            
        return new_embed if new_embed is not None else embeds[start_idx]
    
    def fuse_and_sum1(self, embeds, start_idx, count):
        # new_embed = embeds[start_idx]
        new_embed = None
        for i in range(start_idx, start_idx + count):
            for j in range(i + 1, start_idx + count): 
                # print(embeds[:, i, :, :].shape)  # torch.Size([4, 49, 1024])
                fusion_embed = self.fusion1(embeds[:, i, :, :] , embeds[:, j, :, :], embeds[:, j, :, :])[0]
                reverse_fusion_embed = self.fusion1(embeds[:, j, :, :], embeds[:, i, :, :], embeds[:, i, :, :])[0]
                if new_embed == None:
                    new_embed = fusion_embed + reverse_fusion_embed
                else:
                    new_embed = new_embed + fusion_embed + reverse_fusion_embed
      
        for i in range(start_idx, start_idx + count):
            if new_embed is None :
                new_embed = embeds[:, i, :, :]
            else:
                new_embed += embeds[:, i, :, :]
            
        return new_embed if new_embed is not None else embeds[start_idx]
    
    def fuse_and_sum2(self, embeds, start_idx, count):
        # new_embed = embeds[start_idx]
        new_embed = None
        for i in range(start_idx, start_idx + count):
            for j in range(i + 1, start_idx + count): 
                # print(embeds[:, i, :, :].shape)  # torch.Size([4, 49, 1024])
                fusion_embed = self.fusion2(embeds[:, i, :, :] , embeds[:, j, :, :], embeds[:, j, :, :])[0]
                reverse_fusion_embed = self.fusion2(embeds[:, j, :, :], embeds[:, i, :, :], embeds[:, i, :, :])[0]
                if new_embed == None:
                    new_embed = fusion_embed + reverse_fusion_embed
                else:
                    new_embed = new_embed + fusion_embed + reverse_fusion_embed
         
        for i in range(start_idx, start_idx + count):
            if new_embed is None :
                new_embed = embeds[:, i, :, :]
            else:
                new_embed += embeds[:, i, :, :]
            
        return new_embed if new_embed is not None else embeds[start_idx]


    def prompt_wrap(self, img_embeds, atts_img):
        prompt=f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        #print(wrapped_atts_img.shape)  # torch.Size([4, 194, 2048])
        #print(wrapped_atts_img.shape) # torch.Size([4, 194])
        return wrapped_img_embeds, wrapped_atts_img
    
    def process_image_sets(self, main_embeds, temp_embeds1, temp_embeds2):
        start_idx = 0  # For main, temp1, temp2
        count = 3
        new_image_embeds = self.fuse_and_sum(main_embeds, start_idx, count)
        embeddins_temp1 = self.fuse_and_sum1(temp_embeds1, start_idx, count)
        embeddins_temp2 = self.fuse_and_sum2(temp_embeds2, start_idx, count)
        # print("STRAT")
        # print(new_image_embeds.shape)  # torch.Size([4, 49, 1024])
        # print(embeddins_temp1.shape) # torch.Size([4, 49, 1024])
        # print(embeddins_temp2.shape) # torch.Size([4, 49, 1024])
        new_embed = torch.cat([new_image_embeds, embeddins_temp1], dim=1) 
        new_embed = torch.cat([new_embed, embeddins_temp2], dim=1) 
        return new_embed

#     return return_data
    def forward(self, samples):
        image = samples["images"]
        temp_images1 = samples["temp_images1"]
        temp_images2 = samples["temp_images2"]
        retr_text = samples["retriver_text"]

        bs = len(retr_text)
        img_embeds, atts_img = self.encode_img(image, bs, temp_images1, temp_images2)
        img_embeds = self.layer_norm(img_embeds)

        query_feature = self.text_image(img_embeds) 
        retr_text_embed, _ = self.encode_text(retr_text, query_feature)  
        retr_text_embed = self.iit_proj(retr_text_embed)
        retr_text_embed = self.iit_layernorm(retr_text_embed)  
        new_embeds = torch.cat([img_embeds, retr_text_embed], dim=2)  
        new_embeds = self.new_embeds_proj(new_embeds) # torch.Size([8, 49, 2048])
        new_embeds = self.new_embeds_layernorm(new_embeds)  
        img_embeds, atts_img = self.prompt_wrap(new_embeds, atts_img)
 
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image[0].device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    
    def validation_step(self, samples, batch_idx):
        
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(  
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=300,
            add_special_tokens=False
        )
        image = samples["images"]
        temp_images1 = samples["temp_images1"]
        temp_images2 = samples["temp_images2"]
        retr_text = samples["retriver_text"]
        bs = len(retr_text)
        img_embeds, atts_img = self.encode_img(image, bs, temp_images1, temp_images2)
        img_embeds = self.layer_norm(img_embeds)

        query_feature = self.text_image(img_embeds) 
        retr_text_embed, _ = self.encode_text(retr_text, query_feature)  
        retr_text_embed = self.iit_proj(retr_text_embed)
        retr_text_embed = self.iit_layernorm(retr_text_embed)  
        new_embeds = torch.cat([img_embeds, retr_text_embed], dim=2)  
        new_embeds = self.new_embeds_proj(new_embeds) # torch.Size([8, 49, 2048])
        new_embeds = self.new_embeds_layernorm(new_embeds)  
        img_embeds, atts_img = self.prompt_wrap(new_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)
        inputs_embeds = inputs_embeds.to(torch.bfloat16)
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )

        self.val_step_outputs.append(outputs)
        self.val_step_ref.append(to_regress_tokens["input_ids"])
        self.val_step_ids.append(self.pad_and_convert_to_tensor(samples["id"]))
        return [outputs, to_regress_tokens["input_ids"], self.pad_and_convert_to_tensor(samples["id"])]
    
    def truncate_texts(self, input_texts, max_tokens=100):
        truncated_texts = []
        for text in input_texts:
            # Split each text by spaces
            tokens = text.split()
            # Truncate to the maximum number of tokens
            truncated_tokens = tokens[:max_tokens]
            # Join the tokens back into a single string with spaces
            truncated_text = ' '.join(truncated_tokens)
            truncated_texts.append(truncated_text)
        
        return truncated_texts
    
    def pad_and_convert_to_tensor(self, id_list, max_length=50):
        # Convert each string ID to bytes and pad to max_length
        byte_ids = [s.encode('utf-8') for s in id_list]
        padded_ids = [b.ljust(max_length, b'\0') for b in byte_ids]
        return torch.tensor([list(b) for b in padded_ids], dtype=torch.uint8)
    
    def convert_bytes_to_string(self, byte_list):
        
        byte_string = bytes(byte_list)
       
        stripped_string = byte_string.rstrip(b'\0')
         
        decoded_string = stripped_string.decode('utf-8')
        return decoded_string
    
    def decode(self, output_token):   
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def setup_distributed(self):
        if not dist.is_initialized():
            dist.init_process_group(backend='gloo')  # Initialize with gloo for CPU

    def merge(self,outputs):
        if dist.is_initialized():
            all_rank_outputs = [None for _ in range(dist.get_world_size())]    
            dist.all_gather_object(all_rank_outputs,outputs)
            outputs = [x for y in all_rank_outputs for x in y] ## all_rank_output[i]: i-th batch output 
        single_batch_output_cnt = len(outputs[0])
        ret = [[] for _ in range(single_batch_output_cnt)]
        for idx in range(single_batch_output_cnt):
            for batch in outputs:
                ret[idx].append(batch[idx])
        return ret
    
    def on_validation_epoch_end(self):
        self.setup_distributed()
        self.trainer.strategy.barrier()
        gathered_outputs = self.merge(self.val_step_outputs)
        gathered_ref = self.merge(self.val_step_ref)
        gathered_ids = self.merge(self.val_step_ids)
        gathered_outputs =  [item for sublist in gathered_outputs for item in sublist]
        gathered_ref = [item for sublist in gathered_ref for item in sublist]
        gathered_ids = [item for sublist in gathered_ids for item in sublist]
        
        if self.trainer.is_global_zero:
            
            ref, hypo, ids = gathered_ref, gathered_outputs, gathered_ids
            ref = self.truncate_texts([self.decode(i) for i in ref])
            
            hypo = [self.decode(i) for i in hypo]
            hypo = [ i.replace('\n', ' ') for i in hypo]
            hypo = [ i.replace('  ', ' ') for i in hypo]
            ids = [self.convert_bytes_to_string(i) for i in ids]
             
            refs = {k: [v] for k, v in zip(ids, ref)}
            hypos = {k: [v] for k, v in zip(ids, hypo)}

            eval_res = self.score(ref=refs, hypo=hypos)
            self.log_dict(eval_res, sync_dist=True, logger=True, rank_zero_only=True)

            result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
            os.makedirs(result_folder, exist_ok=True)
            current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
            json.dump(hypos, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
            json.dump(refs, open(os.path.join(result_folder, f'refs_{current_epoch}_{global_step}.json'), 'w'))
            self.print(eval_res)

            self.val_step_outputs.clear()
            self.val_step_ids.clear()
            self.val_step_ref.clear()
            self.save_checkpoint(eval_res)


    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(  
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=300,
            add_special_tokens=False
        )
        
        retr_text = samples["retriver_text"] 
        image = samples["images"]
        temp_images1 = samples["temp_images1"]
        temp_images2 = samples["temp_images2"]

        bs = len(retr_text)
        img_embeds, atts_img = self.encode_img(image, bs, temp_images1, temp_images2)
        img_embeds = self.layer_norm(img_embeds)

        query_feature = self.text_image(img_embeds)#####  
        retr_text_embed, _ = self.encode_text(retr_text, query_feature)  
        retr_text_embed = self.iit_proj(retr_text_embed)
        retr_text_embed = self.iit_layernorm(retr_text_embed)  
        new_embeds = torch.cat([img_embeds, retr_text_embed], dim=2)  
        new_embeds = self.new_embeds_proj(new_embeds) # torch.Size([8, 49, 2048])
        new_embeds = self.new_embeds_layernorm(new_embeds)  
        img_embeds, atts_img = self.prompt_wrap(new_embeds, atts_img)
        
        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)
        inputs_embeds = inputs_embeds.to(torch.bfloat16)
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )

        self.val_step_outputs.append(outputs)
        self.val_step_ref.append(to_regress_tokens["input_ids"])
        self.val_step_ids.append(self.pad_and_convert_to_tensor(samples["id"]))
        return [outputs, to_regress_tokens["input_ids"], self.pad_and_convert_to_tensor(samples["id"])]


    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        self.setup_distributed()
        self.trainer.strategy.barrier()
        gathered_outputs = self.merge(self.val_step_outputs)
        gathered_ref = self.merge(self.val_step_ref)
        gathered_ids = self.merge(self.val_step_ids)
        gathered_outputs =  [item for sublist in gathered_outputs for item in sublist]
        gathered_ref = [item for sublist in gathered_ref for item in sublist]
        gathered_ids = [item for sublist in gathered_ids for item in sublist]
        
        if self.trainer.is_global_zero:
            
            ref, hypo, ids = gathered_ref, gathered_outputs, gathered_ids
            ref = self.truncate_texts([self.decode(i) for i in ref])
            hypo = [self.decode(i) for i in hypo]
            ids = [self.convert_bytes_to_string(i) for i in ids]
             
            refs = {k: [v] for k, v in zip(ids, ref)}
            hypos = {k: [v] for k, v in zip(ids, hypo)}

            eval_res = self.score(ref=refs, hypo=hypos)
            self.log_dict(eval_res, sync_dist=True, logger=True, rank_zero_only=True)

            result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
            os.makedirs(result_folder, exist_ok=True)
            current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
            json.dump(hypos, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
            json.dump(refs, open(os.path.join(result_folder, f'refs_{current_epoch}_{global_step}.json'), 'w'))
            self.print(eval_res)

            self.val_step_outputs.clear()
            self.val_step_ids.clear()
            self.val_step_ref.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
        