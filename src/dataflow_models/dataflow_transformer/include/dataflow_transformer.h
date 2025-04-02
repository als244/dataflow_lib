#ifndef DATAFLOW_TRANSFORMER_H
#define DATAFLOW_TRANSFORMER_H

#include "dataflow_handle.h"
#include "dataflow_models.h"
#include "dataflow_ops.h"

#include "transformer_structs.h"


Transformer_Block * init_transformer_block(DataflowDatatype block_dt, DataflowDatatype compute_dt,
						   DataflowNormalizationType normalization_type, 
						   DataflowAttentionType attention_type,
						   DataflowMLPType mlp_type,
						   DataflowActivationType activation_type,
						   float eps, int theta,
						   int num_q_heads, int num_kv_heads, int head_dim,
						   int ffn_dim,
						   MoE_Config * moe_config,
						   int pointer_alignment);

uint64_t get_transformer_block_raw_size(Transformer_Block * transformer_block);

uint64_t get_transformer_block_aligned_size(Transformer_Block * transformer_block);


// now pass in a buffer of size >= size specified above
// and the pointers will be properly assigned (ensuring alignment)
int bind_transformer_block(void * buffer, Transformer_Block * transformer_block);


// the file consists of all weights 
//int load_transformer_block(char * filename, Transformer_Block * transformer_block);


// Need to set Seq Batch metadata...!
//int bind_transformer_block_activations(void * buffer, Seq_Batch * seq_batch, Transformer_Block * block, Transformer_Block_Activations * activation_buffer);





int submit_transformer_block(Dataflow_Handle * dataflow_handle, int compute_stream_id, void * X, Transformer_Block * transformer_block, Transformer_Block_Activations * activations);

int submit_transformer_head(Dataflow_Handle * dataflow_handle, int compute_stream_id,
                        void * X, Transformer_Head * transformer_head,
                        Transformer_Head_Activations * head_activations, 
						void * logits_out);


int submit_transformer_block_bwd_x(Dataflow_Handle * dataflow_handle, int compute_stream_id, void * dX_out, void * X,Transformer_Block * transformer_block, Transformer_Block_Activations * activations, Transformer_Block_Activations * grad_activations);

int submit_transformer_block_bwd_w(Dataflow_Handle * dataflow_handle, int compute_stream_id, void * dX_out, Transformer_Block_Activations * activations, Transformer_Block_Activations * grad_activations, Transformer_Block * grad_weights);

int submit_transformer_head_bwd_x(Dataflow_Handle * dataflow_handle, int compute_stream_id,
                        void * logits_out, uint32_t * labels, Transformer_Head * transformer_head,
                        Transformer_Head_Activations * head_activations,
                        Transformer_Head_Activations * grad_head_activations,
                        void * dX_out);

int submit_transformer_head_bwd_w(Dataflow_Handle * dataflow_handle, int compute_stream_id,
                         Transformer_Head_Activations * head_activations,
                         Transformer_Head_Activations * grad_head_activations,
                         Transformer_Head * grad_transformer_head);

#endif