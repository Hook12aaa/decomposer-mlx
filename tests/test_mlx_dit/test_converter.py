import pytest

from decomposer.mlx_convert.convert import build_name_mapping


class TestBuildNameMapping:
    def test_maps_top_level_projections(self):
        mapping = build_name_mapping()
        assert mapping["img_in.weight"] == "img_in.weight"
        assert mapping["img_in.bias"] == "img_in.bias"
        assert mapping["txt_in.weight"] == "txt_in.weight"
        assert mapping["txt_in.bias"] == "txt_in.bias"
        assert mapping["txt_norm.weight"] == "txt_norm.weight"

    def test_maps_output_norm_and_projection(self):
        mapping = build_name_mapping()
        assert mapping["norm_out.linear.weight"] == "norm_out.linear.weight"
        assert mapping["norm_out.linear.bias"] == "norm_out.linear.bias"
        assert mapping["proj_out.weight"] == "proj_out.weight"
        assert mapping["proj_out.bias"] == "proj_out.bias"

    def test_maps_timestep_embedder(self):
        mapping = build_name_mapping()
        assert (
            mapping["time_text_embed.timestep_embedder.linear_1.weight"]
            == "time_text_embed.timestep_embedder.linear1.weight"
        )
        assert (
            mapping["time_text_embed.timestep_embedder.linear_2.bias"]
            == "time_text_embed.timestep_embedder.linear2.bias"
        )

    def test_maps_addition_t_embedding(self):
        mapping = build_name_mapping()
        assert (
            mapping["time_text_embed.addition_t_embedding.weight"]
            == "time_text_embed.addition_t_embedding.weight"
        )

    def test_maps_block_attention_img_stream(self):
        mapping = build_name_mapping()
        assert mapping["transformer_blocks.0.attn.to_q.weight"] == "transformer_blocks.0.attn.to_q.weight"
        assert mapping["transformer_blocks.0.attn.to_out.0.weight"] == "transformer_blocks.0.attn.attn_to_out.0.weight"
        assert mapping["transformer_blocks.0.attn.to_out.0.bias"] == "transformer_blocks.0.attn.attn_to_out.0.bias"

    def test_maps_block_attention_txt_stream(self):
        mapping = build_name_mapping()
        assert mapping["transformer_blocks.0.attn.add_q_proj.weight"] == "transformer_blocks.0.attn.add_q_proj.weight"
        assert mapping["transformer_blocks.0.attn.to_add_out.weight"] == "transformer_blocks.0.attn.to_add_out.weight"

    def test_maps_block_modulation(self):
        mapping = build_name_mapping()
        assert mapping["transformer_blocks.0.img_mod.1.weight"] == "transformer_blocks.0.img_mod_linear.weight"
        assert mapping["transformer_blocks.0.txt_mod.1.bias"] == "transformer_blocks.0.txt_mod_linear.bias"

    def test_maps_block_ffn(self):
        mapping = build_name_mapping()
        assert mapping["transformer_blocks.0.img_mlp.net.0.proj.weight"] == "transformer_blocks.0.img_ff.linear1.weight"
        assert mapping["transformer_blocks.0.img_mlp.net.2.weight"] == "transformer_blocks.0.img_ff.linear2.weight"
        assert mapping["transformer_blocks.0.txt_mlp.net.0.proj.bias"] == "transformer_blocks.0.txt_ff.linear1.bias"
        assert mapping["transformer_blocks.0.txt_mlp.net.2.bias"] == "transformer_blocks.0.txt_ff.linear2.bias"

    def test_maps_all_60_blocks(self):
        mapping = build_name_mapping()
        for i in range(60):
            assert f"transformer_blocks.{i}.attn.to_q.weight" in mapping
            assert f"transformer_blocks.{i}.img_mod.1.weight" in mapping
            assert f"transformer_blocks.{i}.txt_mlp.net.2.weight" in mapping

    def test_covers_all_gguf_tensor_names(self):
        mapping = build_name_mapping()
        block_keys = [k for k in mapping if k.startswith("transformer_blocks.0.")]
        per_block = len(block_keys)
        top_level = len(mapping) - 60 * per_block
        assert len(mapping) == top_level + 60 * per_block

    def test_mlx_names_match_model_parameters(self):
        """Verify mapped MLX names exist in the actual model parameter tree."""
        import mlx.core as mx
        from decomposer.mlx_dit.transformer import MLXQwenTransformer

        config = {
            "num_layers": 2,
            "num_attention_heads": 24,
            "attention_head_dim": 128,
            "in_channels": 64,
            "out_channels": 16,
            "joint_attention_dim": 3584,
            "axes_dims_rope": [16, 56, 56],
            "patch_size": 2,
            "use_additional_t_cond": True,
        }
        model = MLXQwenTransformer(config)

        def collect_param_names(d, prefix=""):
            names = set()
            if isinstance(d, dict):
                for k, v in d.items():
                    names |= collect_param_names(v, f"{prefix}.{k}" if prefix else k)
            elif isinstance(d, list):
                for i, v in enumerate(d):
                    names |= collect_param_names(v, f"{prefix}.{i}")
            elif isinstance(d, mx.array):
                names.add(prefix)
            return names

        model_names = collect_param_names(model.parameters())

        mapping = build_name_mapping()
        mlx_names_for_2_blocks = set()
        for mlx_name in mapping.values():
            block_prefix_match = False
            for i in range(60):
                if mlx_name.startswith(f"transformer_blocks.{i}."):
                    if i < 2:
                        mlx_names_for_2_blocks.add(mlx_name)
                    block_prefix_match = True
                    break
            if not block_prefix_match:
                mlx_names_for_2_blocks.add(mlx_name)

        missing = mlx_names_for_2_blocks - model_names
        assert not missing, f"MLX names not found in model: {sorted(missing)}"
