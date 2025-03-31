#ifndef DATAFLOW_TRANSFORMER_H
#define DATAFLOW_TRANSFORMER_H

#include "dataflow_handle.h"
#include "dataflow_models.h"
#include "dataflow_ops.h"


typedef struct transformer_block_weight_offsets {
	uint64_t w_attn_norm;
	uint64_t w_q;
	uint64_t w_k;
	uint64_t w_v;
	uint64_t w_o;
	uint64_t w_ffn_norm;
	uint64_t w_router;
	// arrays of length num_local experts if DATAFLOW_MOE_MLP
	// otherwise arrays of length 1
	uint64_t * w_1;
	uint64_t * w_2;
	uint64_t * w_3;
} Transformer_Block_Weight_Offsets;

typedef struct transformer_block_config {
	DataflowDatatype block_dt;
	DataflowNormalizationType normalization_type;
	DataflowAttentionType attention_type;
	DataflowMLPType mlp_type;
	DataflowActivationType activation_type;
	int num_q_heads;
	int num_kv_heads;
	int head_dim;

	// We set these
	// model_dim = num_q_heads * head_dim
	// kv_dim = num_kv_heads * head_dim
	int model_dim;
	int kv_dim;

	// if using mlp_type = DATAFLOW_MOE_MLP
	// this ffn_dim is the dimension of each expert
	int ffn_dim;

	// for models with mlp_type = DATAFLOW_MOE_MLP
	MoE_Config moe_config;
		
	// each unqiue weight pointer
	// must be a multiple of this value
	// (e.g. 256 to use tensor cores, or maybe 512 to load direct from SSD, or maybe 4k to start on unique sys ram pages, etc...)
	int pointer_alignment;

	Transformer_Block_Weight_Offsets weight_offsets;
	// the total amount of memory required
	// to hold all weights
	uint64_t block_raw_size;
	
	// the total amount of memory
	uint64_t block_aligned_size;
	
} Transformer_Block_Config;

typedef struct transformer_block {
	Transformer_Block_Config config;
	void * buffer;
	// offsets into buffer, based upon
	// config.weight_offsets
	void * w_attn_norm;
	void * w_q;
	void * w_k;
	void * w_v;
	void * w_o;
	void * w_ffn_norm;
	// if non-moe this is null
	void * w_router;
	// each of these 
	// are arrays of length = MAX(config.num_local_experts, 1)
	// (if non-moe, just use index 0)

	// if DATAFLOW_VANILLA_MLP, then w_3 should all be null
	void ** w_1;
	void ** w_2;
	void ** w_3;
} Transformer_Block;


typedef struct transformer_block_activations {
	Transformer_Block * block;
	// for rms norm: weighted_sums & rms_vals
	void * buffer;
	void * x_attn_norm_misc;
	void * x_q;
	// Set x_k and x_v
	// to be within seq context found
	// within seq_batch
	void * x_k;
	void * x_v;
	// softmax_lse
	void * x_attn_misc;
	void * x_o;
	// for rms norm: weighted_sums & rms_vals
	void * x_ffn_norm_misc;
	void ** w_1;
	void ** w_2;
	void ** w_3;
} Transformer_Block_Activations;


Transformer_Block * init_transformer_block(DataflowDatatype block_dt,
						   DataflowNormalizationType normalization_type, 
						   DataflowAttentionType attention_type,
						   DataflowMLPType mlp_type,
						   DataflowActivationType activation_type,
						   int num_q_heads, int num_kv_heads, int head_dim,
						   int ffn_dim,
						   MoE_Config * moe_config,
						   int pointer_alignment);

// first init must be called
uint64_t get_transformer_block_raw_size(Transformer_Block * transformer_block);
uint64_t get_transformer_block_aligned_size(Transformer_Block * transformer_block);

// now pass in a buffer of size >= size specified above
// and the pointers will be properly assigned (ensuring alignment)
int bind_transformer_block(void * buffer, Transformer_Block * transformer_block);


// the file consists of all weights 
//int load_transformer_block(char * filename, Transformer_Block * transformer_block);


//int bind_transformer_block_activations(void * buffer, Seq_Batch * seq_batch, Transformer_Block * block, Transformer_Block_Activations * activation_buffer);


//int submit_transformer_block(Seq_Batch * seq_batch, Transformer_Block * transformer_block, Transformer_Block_Activations * activation_buffer);


#endif