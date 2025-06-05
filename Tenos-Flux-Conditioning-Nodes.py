# --- START OF FILE tenosai_flux_conditioning.py ---
# This file contains Tenos.ai nodes for advanced Flux Dev conditioning.
# These nodes are updated to correctly handle layered CONDITIONING inputs
# and apply logic based on the 'source_layer' tag in input dictionaries.

import torch
import comfy.conds
from server import PromptServer # Import for sending feedback to the UI

# Helper function to normalize a tensor
def normalize_tensor(tensor):
    if tensor is None:
        return None
    norm = torch.linalg.norm(tensor, dim=-1, keepdim=True)
    # Handle zero norm case to prevent division by zero
    # Avoid modifying in-place if possible, return new tensor
    return torch.where(norm > 1e-6, tensor / norm, tensor.new_zeros(tensor.shape, device=tensor.device, dtype=tensor.dtype))

# Helper function to safely validate and extract layered conditioning inputs
# Returns a list of [tensor, dict] tuples and any errors found.
def _validate_and_extract_layered_conditioning(conditioning_list, input_name):
    input_errors = []
    extracted_layers = []

    if not isinstance(conditioning_list, list) or len(conditioning_list) == 0:
        input_errors.append(f"Input '{input_name}' is required but is empty or not a valid list format. Expected list of [tensor, dict].")
        return [], input_errors # Return empty list and errors

    for i, item in enumerate(conditioning_list):
        if not isinstance(item, list) or len(item) < 2:
            input_errors.append(f"Input '{input_name}' item {i} has invalid format. Expected [tensor, dict], got {type(item)}.")
            continue # Skip this invalid item but continue processing others

        tensor = item[0]
        dictionary = item[1] if isinstance(item[1], dict) else {} # Ensure dictionary is a dict

        if not isinstance(tensor, torch.Tensor):
            input_errors.append(f"Input '{input_name}' item {i} has invalid tensor type. Expected torch.Tensor, got {type(tensor)}.")
            continue # Skip this invalid item

        # Store the extracted tensor and dictionary
        extracted_layers.append([tensor, dictionary])

    if not extracted_layers and not input_errors:
         # This case means the input list was not empty, but all items were invalid/skipped
         input_errors.append(f"Input '{input_name}' was provided but contained no valid [tensor, dict] items after format check.")

    return extracted_layers, input_errors

# Helper to pad tensors in a list to the maximum sequence length
# Assumes tensors have shape [Batch, SeqLen, EmbeddingDim] and share Batch/EmbeddingDim
def _pad_tensors_to_max_seqlen(tensor_list):
    if not tensor_list:
        return []

    max_seq_len = max(t.shape[1] for t in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        if tensor.shape[1] < max_seq_len:
            padding = (0, 0, 0, max_seq_len - tensor.shape[1]) # Pad (left_emb, right_emb, top_seq, bottom_seq)
            padded_tensor = torch.nn.functional.pad(tensor, padding)
            padded_tensors.append(padded_tensor)
        else:
            padded_tensors.append(tensor)
    return padded_tensors

# Helper to aggregate (sum) tensors belonging to a specific source_layer tag, plus untagged layers
# Returns a single tensor (sum of matching and untagged layers), or a zero tensor if none found.
def _aggregate_layers(layered_conditioning_data, target_layer_tag, max_seq_len, embedding_dim, device, dtype):
    matching_tensors = []
    untagged_tensors = []

    for tensor, dictionary in layered_conditioning_data:
        current_layer_tag = dictionary.get("source_layer")
        # Ensure padding is applied before aggregation
        # Pad here first, then sum
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, max_seq_len - tensor.shape[1])) # Pad SeqLen dim

        if current_layer_tag == target_layer_tag:
            matching_tensors.append(padded_tensor)
        elif current_layer_tag is None: # Layers without a 'source_layer' tag contribute to all blocks
             untagged_tensors.append(padded_tensor)

    # Aggregate matching layers
    aggregated_tensor = None
    if matching_tensors:
        aggregated_tensor = torch.stack(matching_tensors).sum(dim=0) # Sum along batch dim for different layers
        # The previous comment was wrong. Sum should be over the list of tensors, result shape is Batch, Seq, Emb.
        # Example: T1[B, S1, E], T2[B, S2, E]. Pad to max(S1, S2). T1_padded[B, maxS, E], T2_padded[B, maxS, E].
        # Stack gives [2, B, maxS, E]. Sum(dim=0) gives [B, maxS, E]. Correct.

    # Aggregate untagged layers
    untagged_aggregated_tensor = None
    if untagged_tensors:
         untagged_aggregated_tensor = torch.stack(untagged_tensors).sum(dim=0)

    # Combine matching and untagged contributions for this block
    if aggregated_tensor is not None and untagged_aggregated_tensor is not None:
         return aggregated_tensor + untagged_aggregated_tensor
    elif aggregated_tensor is not None:
         return aggregated_tensor
    elif untagged_aggregated_tensor is not None:
         return untagged_aggregated_tensor
    else:
        # Return a zero tensor of the correct shape if no layers matched or were untagged
        # Need batch size from input. Assume all inputs share the same batch size (validated earlier).
        batch_size = layered_conditioning_data[0][0].shape[0] if layered_conditioning_data else 1 # Default to 1 if no inputs
        return torch.zeros((batch_size, max_seq_len, embedding_dim), device=device, dtype=dtype)


# Helper to aggregate (sum) ALL tensors regardless of source_layer tag
def _aggregate_all_layers(layered_conditioning_data, max_seq_len, embedding_dim, device, dtype):
    all_tensors = []
    if not layered_conditioning_data:
         # Return a zero tensor if no input data
         batch_size = 1 # Default batch size
         return torch.zeros((batch_size, max_seq_len, embedding_dim), device=device, dtype=dtype)

    batch_size = layered_conditioning_data[0][0].shape[0] # Get batch size from the first valid tensor

    for tensor, dictionary in layered_conditioning_data:
         # Ensure padding is applied before aggregation
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, max_seq_len - tensor.shape[1])) # Pad SeqLen dim
        all_tensors.append(padded_tensor)

    if not all_tensors:
         # This case should be caught by the initial check, but as a fallback
         return torch.zeros((batch_size, max_seq_len, embedding_dim), device=device, dtype=dtype)

    return torch.stack(all_tensors).sum(dim=0)


# Helper function to prepare a single output conditioning item [tensor, dict]
# Copies base_dict and adds/overwrites source_layer.
def _prepare_output_item(tensor, base_dict, source_layer=None):
    output_dict = base_dict.copy()
    if source_layer is not None:
        output_dict["source_layer"] = source_layer
    return [tensor, output_dict]

# --- Node 1: TenosaiFluxConditioningApply (Single Input, Layered Output) ---
class TenosaiFluxConditioningApply:
    """
    FLUX Dev ONLY: Applies individual strength multipliers to each layer
    of a single CONDITIONING input based on its 'source_layer' tag
    or to untagged layers for all blocks. Generates layered output where
    each original input layer contributes to scaled output layers per block.
    Preserves original conditioning data like 'pooled_output'.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                # Accept layered conditioning input
                "conditioning": ("CONDITIONING", {"display": "conditioning", "tooltip": "The input conditioning, potentially containing multiple layers ([tensor, dict] tuples). Original dictionary data is preserved."}),

                # Option to normalize each input tensor BEFORE applying block strengths
                "normalize_input_layers": ("BOOLEAN", {"default": False, "display": "boolean", "tooltip": "Normalize each individual input tensor before applying block strengths"}),

                # Strength controls for applying the single input tensor to specific Flux diffusion blocks (0.0-10.0+ range).
                "encoder_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Multiplier for contributions to 'encoder' block output (Input Blocks 0-5)"}),
                "base_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Multiplier for contributions to 'base' block output (Input Blocks 6-18 + Middle)"}),
                "decoder_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Multiplier for contributions to 'decoder' block output (Output Blocks 0-37)"}),
                "out_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Multiplier for contributions to 'out' (final layer) block"}),
            },
            "optional": {},
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
        }
        return inputs

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("CONDITIONING", "apply_summary")
    FUNCTION = "process_conditioning"
    CATEGORY = "Tenos.ai/Conditioning"
    DESCRIPTION = "FLUX Dev ONLY: Applies multipliers to each input conditioning layer for specific Flux blocks, outputs layered conditioning."

    def process_conditioning(self, conditioning, normalize_input_layers,
                             encoder_multiplier, base_multiplier, decoder_multiplier, out_multiplier,
                             unique_id=None, extra_pnginfo=None):

        # --- Input Collection and Validation ---
        input_layers, input_errors = _validate_and_extract_layered_conditioning(conditioning, "conditioning")

        if not input_layers:
            summary = "No valid conditioning layers found in input. Returning empty conditioning."
            if input_errors: summary += f"\nInput Errors/Warnings ({len(input_errors)}):\n" + "\n".join(input_errors)
            print(f"TenosaiFluxConditioningApply (Node ID {unique_id}): {summary}")
            if unique_id is not None: self._send_feedback(unique_id, "apply_summary", summary)
            return ([], summary)

        # Determine max sequence length, batch size, and embedding dim from valid inputs
        max_seq_len = max(layer[0].shape[1] for layer in input_layers)
        batch_size = input_layers[0][0].shape[0] # Assume consistent batch size
        embedding_dim = input_layers[0][0].shape[2] # Assume consistent embedding dim
        device = input_layers[0][0].device
        dtype = input_layers[0][0].dtype

        # Pad all input tensors to the max sequence length
        # Also apply input normalization per layer
        processed_input_layers = []
        for tensor, dictionary in input_layers:
             current_tensor = tensor # Use a mutable variable name
             if normalize_input_layers:
                  current_tensor = normalize_tensor(current_tensor)
             # Pad AFTER normalization
             padded_tensor = torch.nn.functional.pad(current_tensor, (0, 0, 0, max_seq_len - current_tensor.shape[1]))
             processed_input_layers.append([padded_tensor, dictionary])


        # --- Apply Strength Controls per Block and Prepare Output ---
        output_conditioning = []
        output_summary_lines = [
            f"Input layers: {len(processed_input_layers)}",
            f"Max sequence length: {max_seq_len}",
            f"Embedding dimension: {embedding_dim}",
            f"Batch size: {batch_size}",
            "--- Block Application ---"
        ]
        if normalize_input_layers: output_summary_lines[0] += " (Input layers normalized)"


        block_multipliers = {
            'encoder': encoder_multiplier,
            'base': base_multiplier,
            'decoder': decoder_multiplier,
            'out': out_multiplier,
        }

        ordered_flux_blocks = ['encoder', 'base', 'decoder', 'out']

        generated_item_count = 0
        # Iterate through each *input layer*
        for input_tensor, input_dict in processed_input_layers:
             # For each input layer, generate scaled contributions for each output block
             for block_name in ordered_flux_blocks:
                  multiplier = block_multipliers.get(block_name, 0.0)

                  # Only generate conditioning item for blocks with non-zero multiplier (use tolerance)
                  if multiplier > 1e-6:
                       # Scale the input tensor by the block's multiplier
                       scaled_tensor = input_tensor * multiplier

                       # Create output item using dictionary from the input layer
                       output_conditioning.append(_prepare_output_item(scaled_tensor.clone(), input_dict, source_layer=block_name))
                       generated_item_count += 1
                       # Report the multiplier in the summary for this input layer and block
                       output_summary_lines.append(f" - InputLayer(seq_len={input_tensor.shape[1]}): Multiplier {multiplier:.4f} for '{block_name}'.")
             if abs(input_dict.get("source_layer", -1)) >= 0 : # Check if source_layer exists and isn't None
                  pass # Already reported per block
             else:
                  output_summary_lines.append(f" - InputLayer(seq_len={input_tensor.shape[1]}, untagged): Applied to all blocks.")


        if not output_conditioning:
             summary = "\n".join(output_summary_lines)
             summary += "\nWarning: No output conditioning generated (all block multipliers were zero or near zero for all input layers)."
             if input_errors: summary += f"\nInput Warnings ({len(input_errors)}):\n" + "\n".join(input_errors)
             print(f"TenosaiFluxConditioningApply (Node ID {unique_id}): {summary}")
             if unique_id is not None: self._send_feedback(unique_id, "apply_summary", summary)
             return ([], summary)

        # --- Final Summary ---
        summary = "\n".join(output_summary_lines)
        summary += f"\nGenerated total {generated_item_count} layered conditioning items."
        if input_errors: summary += f"\nInput Warnings ({len(input_errors)}):\n" + "\n".join(input_errors)

        # Print final summary to console
        print(f"TenosaiFluxConditioningApply (Node ID {unique_id}):\n{summary}")

        # Send feedback via PromptServer to the UI, targeting the output string widget
        if unique_id is not None:
             self._send_feedback(unique_id, "apply_summary", summary)

        # Return the final combined conditioning list (layered) and the summary string
        return (output_conditioning, summary)

    def _send_feedback(self, unique_id, widget_name, message):
         """Helper to send feedback to the ComfyUI frontend using PromptServer."""
         node_id_str = str(unique_id)
         try:
              if hasattr(PromptServer.instance, 'send_sync'):
                   PromptServer.instance.send_sync("impact-node-feedback", {
                       "node_id": node_id_str,
                       "widget_name": widget_name,
                       "type": "TEXT",
                       "value": message
                   })
              else:
                   pass
         except Exception as e:
              print(f"TenosaiFluxConditioningApply (Node ID {node_id_str}): Failed to send feedback via PromptServer: {e}")

# --- Node 2: TenosaiFluxConditioningBlend (Blends 2 inputs per block, layered output) ---
class TenosaiFluxConditioningBlend:
    """
    FLUX Dev ONLY: Blends two CONDITIONING inputs (Cond1 = base, Cond2 = modifier)
    per block, producing a layered CONDITIONING output. It aggregates layers
    from each input with matching 'source_layer' tags (and untagged layers)
    for each block, blends the aggregates using block-specific ratios, and outputs
    a layered conditioning list with one item per block. Preserves dictionary
    data from the first item of Conditioning 1.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "conditioning_to": ("CONDITIONING", {"display": "conditioning_to", "tooltip": "Base conditioning source (Cond1). Aggregated per block. Original dictionary data preserved from its first valid item."}),
                "conditioning_from": ("CONDITIONING", {"display": "conditioning_from", "tooltip": "Modifying conditioning source (Cond2). Aggregated per block."}),

                "encoder_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Blend ratio (Cond2 contribution) for 'encoder' block (Input Blocks 0-5). 0.0=Cond1 aggregate, 1.0=Cond2 aggregate. Range >1.0 allows extrapolation."}),
                "base_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Blend ratio (Cond2 contribution) for 'base' block (Input Blocks 6-18 + Middle). 0.0=Cond1 aggregate, 1.0=Cond2 aggregate. Range >1.0 allows extrapolation."}),
                "decoder_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Blend ratio (Cond2 contribution) for 'decoder' block (Output Blocks 0-37). 0.0=Cond1 aggregate, 1.0=Cond2 aggregate. Range >1.0 allows extrapolation."}),
                "out_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Blend ratio (Cond2 contribution) for 'out' (final layer) block. 0.0=Cond1 aggregate, 1.0=Cond2 aggregate. Range >1.0 allows extrapolation."}),

                "normalize_to_input_layers": ("BOOLEAN", {"default": False, "display": "boolean", "tooltip": "Normalize each individual tensor from 'conditioning_to' BEFORE aggregation per block."}), # Added layer-wise input norm
                "normalize_from_input_layers": ("BOOLEAN", {"default": False, "display": "boolean", "tooltip": "Normalize each individual tensor from 'conditioning_from' BEFORE aggregation per block."}), # Added layer-wise input norm
                "normalize_per_block_output": ("BOOLEAN", {"default": False, "display": "boolean", "tooltip": "Normalize the resulting blended tensor for each block output item AFTER blending."}),

                "clamp_ratios": ("BOOLEAN", {"default": False, "display": "boolean", "tooltip": "If checked, clamp the effective blend ratio for each block to the range [0.0, 1.0] before blending."}),
            },
            "optional": {},
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
        }
        return inputs

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("CONDITIONING", "blend_summary")
    FUNCTION = "process_conditioning"
    CATEGORY = "Tenos.ai/Conditioning"
    DESCRIPTION = "FLUX Dev ONLY: Blends two CONDITIONING inputs per block, aggregating matching layers (and untagged), outputs layered conditioning."

    def process_conditioning(self, conditioning_to, conditioning_from,
                             encoder_ratio, base_ratio, decoder_ratio, out_ratio,
                             normalize_to_input_layers, normalize_from_input_layers, normalize_per_block_output,
                             clamp_ratios, unique_id=None, extra_pnginfo=None):

        # --- Input Collection and Validation ---
        cond_to_layers, errors_to = _validate_and_extract_layered_conditioning(conditioning_to, "conditioning_to")
        cond_from_layers, errors_from = _validate_and_extract_layered_conditioning(conditioning_from, "conditioning_from")
        input_errors = errors_to + errors_from

        if not cond_to_layers or not cond_from_layers:
             summary = "One or both required conditioning inputs are missing or invalid. Cannot blend. Returning empty conditioning."
             if input_errors: summary += f"\nInput Errors/Warnings ({len(input_errors)}):\n" + "\n".join(input_errors)
             print(f"TenosaiFluxConditioningBlend (Node ID {unique_id}): {summary}")
             if unique_id is not None: self._send_feedback(unique_id, "blend_summary", summary)
             return ([], summary)

        # Determine shapes and device from valid inputs
        all_input_layers = cond_to_layers + cond_from_layers
        max_seq_len = max(layer[0].shape[1] for layer in all_input_layers)
        batch_size = all_input_layers[0][0].shape[0]
        embedding_dim = all_input_layers[0][0].shape[2]
        device = all_input_layers[0][0].device
        dtype = all_input_layers[0][0].dtype

        # Check if batch size and embedding dim match across ALL inputs
        for tensor, dictionary in all_input_layers:
            if tensor.shape[0] != batch_size or tensor.shape[2] != embedding_dim:
                input_errors.append(f"Batch size ({tensor.shape[0]} vs {batch_size}) or Embedding dimension ({tensor.shape[2]} vs {embedding_dim}) mismatch found in an input layer. Cannot blend. Returning empty conditioning.")
                summary = "Shape mismatch in input layers. Cannot blend. Returning empty conditioning."
                if input_errors: summary += f"\nInput Errors/Warnings ({len(input_errors)}):\n" + "\n".join(input_errors)
                print(f"TenosaiFluxConditioningBlend (Node ID {unique_id}): {summary}")
                if unique_id is not None: self._send_feedback(unique_id, "blend_summary", summary)
                return ([], summary)

        # Apply layer-wise input normalizations AFTER validation and BEFORE padding/aggregation
        if normalize_to_input_layers:
             cond_to_layers = [[normalize_tensor(t), d] for t, d in cond_to_layers]
        if normalize_from_input_layers:
             cond_from_layers = [[normalize_tensor(t), d] for t, d in cond_from_layers]


        # --- Aggregate, Blend per Block, and Prepare Output ---
        output_conditioning = []
        output_summary_lines = [
            f"Cond1 layers: {len(cond_to_layers)}, Cond2 layers: {len(cond_from_layers)}",
            f"Max sequence length: {max_seq_len}",
            f"Embedding dimension: {embedding_dim}",
            f"Batch size: {batch_size}",
            "--- Block Blending & Layered Output ---"
        ]
        if normalize_to_input_layers: output_summary_lines[0] += " (Cond1 layers normalized)"
        if normalize_from_input_layers: output_summary_lines[0] += " (Cond2 layers normalized)"

        block_ratios = {
            'encoder': encoder_ratio,
            'base': base_ratio,
            'decoder': decoder_ratio,
            'out': out_ratio,
        }

        ordered_flux_blocks = ['encoder', 'base', 'decoder', 'out']

        generated_block_count = 0
        # Use the dictionary from the first valid 'to' item as the base for output dictionaries
        base_output_dict = cond_to_layers[0][1].copy() if cond_to_layers else {}

        for block_name in ordered_flux_blocks:
            # Aggregate tensors for this block from both inputs
            agg_to_tensor = _aggregate_layers(cond_to_layers, block_name, max_seq_len, embedding_dim, device, dtype)
            agg_from_tensor = _aggregate_layers(cond_from_layers, block_name, max_seq_len, embedding_dim, device, dtype)

            # Get the blend strength value for this output block name
            blend_ratio = block_ratios.get(block_name, 0.0)

            effective_ratio = blend_ratio
            if clamp_ratios:
                 effective_ratio = max(0.0, min(1.0, blend_ratio))
                 if abs(effective_ratio - blend_ratio) > 1e-6:
                      output_summary_lines.append(f" - '{block_name}': Ratio {blend_ratio:.4f} clamped to {effective_ratio:.4f}.")

            # --- Perform Blending for this specific block ---
            # Formula: agg_to * (1.0 - effective_ratio) + agg_from * effective_ratio
            blended_block_tensor = agg_to_tensor * (1.0 - effective_ratio) + agg_from_tensor * effective_ratio

            # Optional Normalization of the resulting tensor for this block
            if normalize_per_block_output:
                blended_block_tensor = normalize_tensor(blended_block_tensor)

            # --- Prepare Output Item for this Block ---
            output_conditioning.append(_prepare_output_item(blended_block_tensor, base_output_dict, source_layer=block_name))
            generated_block_count += 1

            # Report the ratio and shapes in the summary
            output_summary_lines.append(f" - '{block_name}': Blended with ratio {effective_ratio:.4f} (Cond2 contribution). Agg To Shape: {agg_to_tensor.shape}, Agg From Shape: {agg_from_tensor.shape}, Output Shape: {blended_block_tensor.shape}.")

        # --- Final Summary ---
        summary = "\n".join(output_summary_lines)
        summary += f"\nGenerated {generated_block_count} layered conditioning items for {len(ordered_flux_blocks)} blocks."
        if input_errors: summary += f"\nInput Warnings ({len(input_errors)}):\n" + "\n".join(input_errors)

        # Print final summary to console
        print(f"TenosaiFluxConditioningBlend (Node ID {unique_id}):\n{summary}")

        # Send feedback via PromptServer to the UI, targeting the output string widget
        if unique_id is not None:
             self._send_feedback(unique_id, "blend_summary", summary)

        # Return the final combined conditioning list (layered output) and the summary string
        return (output_conditioning, summary)

    def _send_feedback(self, unique_id, widget_name, message):
         """Helper to send feedback to the ComfyUI frontend using PromptServer."""
         node_id_str = str(unique_id)
         try:
              if hasattr(PromptServer.instance, 'send_sync'):
                   PromptServer.instance.send_sync("impact-node-feedback", {
                       "node_id": node_id_str,
                       "widget_name": widget_name,
                       "type": "TEXT",
                       "value": message
                   })
              else:
                   pass
         except Exception as e:
              print(f"TenosaiFluxConditioningBlend (Node ID {node_id_str}): Failed to send feedback via PromptServer: {e}")

# --- Node 3: TenosaiFluxConditioningSummedBlend (Blends 2 inputs per block, sums blended results, outputs single) ---
class TenosaiFluxConditioningSummedBlend:
    """
    FLUX Dev ONLY: Blends two CONDITIONING inputs (Cond1 = base, Cond2 = modifier)
    per block, then sums the resulting blended block tensors into a single output
    tensor. It aggregates layers from each input with matching 'source_layer' tags
    (and untagged layers) for each block, blends the aggregates using block-specific
    ratios, and sums the resulting tensors for all blocks. Preserves dictionary
    data from the first item of Conditioning 1. Outputs single CONDITIONING item.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "conditioning_to": ("CONDITIONING", {"display": "conditioning_to", "tooltip": "Base conditioning source (Cond1). Aggregated per block. Original dictionary data preserved from its first valid item."}),
                "conditioning_from": ("CONDITIONING", {"display": "conditioning_from", "tooltip": "Modifying conditioning source (Cond2). Aggregated per block."}),

                "encoder_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Blend ratio (Cond2 contribution) for 'encoder' block aggregate. 0.0=Cond1 aggregate, 1.0=Cond2 aggregate. Range >1.0 allows extrapolation."}),
                "base_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Blend ratio (Cond2 contribution) for 'base' block aggregate. 0.0=Cond1 aggregate, 1.0=Cond2 aggregate. Range >1.0 allows extrapolation."}),
                "decoder_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Blend ratio (Cond2 contribution) for 'decoder' block aggregate. 0.0=Cond1 aggregate, 1.0=Cond2 aggregate. Range >1.0 allows extrapolation."}),
                "out_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Blend ratio (Cond2 contribution) for 'out' block aggregate. 0.0=Cond1 aggregate, 1.0=Cond2 aggregate. Range >1.0 allows extrapolation."}),

                "normalize_to_input_layers": ("BOOLEAN", {"default": False, "display": "boolean", "tooltip": "Normalize each individual tensor from 'conditioning_to' BEFORE aggregation per block."}),
                "normalize_from_input_layers": ("BOOLEAN", {"default": False, "display": "boolean", "tooltip": "Normalize each individual tensor from 'conditioning_from' BEFORE aggregation per block."}),
                "normalize_output": ("BOOLEAN", {"default": False, "display": "boolean", "tooltip": "Normalize the final summed output tensor."}),

                "clamp_ratios": ("BOOLEAN", {"default": False, "display": "boolean", "tooltip": "If checked, clamp the effective blend ratio for each block to the range [0.0, 1.0] before blending."}),
            },
            "optional": {},
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
        }
        return inputs

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("CONDITIONING", "summed_blend_summary")
    FUNCTION = "process_conditioning"
    CATEGORY = "Tenos.ai/Conditioning"
    DESCRIPTION = "FLUX Dev ONLY: Blends two CONDITIONING inputs per block (aggregating matching layers), sums blended results, outputs single item."

    def process_conditioning(self, conditioning_to, conditioning_from,
                             encoder_ratio, base_ratio, decoder_ratio, out_ratio,
                             normalize_to_input_layers, normalize_from_input_layers, normalize_output,
                             clamp_ratios, unique_id=None, extra_pnginfo=None):

        # --- Input Collection and Validation ---
        cond_to_layers, errors_to = _validate_and_extract_layered_conditioning(conditioning_to, "conditioning_to")
        cond_from_layers, errors_from = _validate_and_extract_layered_conditioning(conditioning_from, "conditioning_from")
        input_errors = errors_to + errors_from

        if not cond_to_layers or not cond_from_layers:
             summary = "One or both required conditioning inputs are missing or invalid. Cannot blend/sum. Returning empty conditioning."
             if input_errors: summary += f"\nInput Errors/Warnings ({len(input_errors)}):\n" + "\n".join(input_errors)
             print(f"TenosaiFluxConditioningSummedBlend (Node ID {unique_id}): {summary}")
             if unique_id is not None: self._send_feedback(unique_id, "summed_blend_summary", summary)
             return ([], summary)

        # Determine shapes and device from valid inputs
        all_input_layers = cond_to_layers + cond_from_layers
        max_seq_len = max(layer[0].shape[1] for layer in all_input_layers)
        batch_size = all_input_layers[0][0].shape[0]
        embedding_dim = all_input_layers[0][0].shape[2]
        device = all_input_layers[0][0].device
        dtype = all_input_layers[0][0].dtype

        # Check if batch size and embedding dim match across ALL inputs
        for tensor, dictionary in all_input_layers:
            if tensor.shape[0] != batch_size or tensor.shape[2] != embedding_dim:
                input_errors.append(f"Batch size ({tensor.shape[0]} vs {batch_size}) or Embedding dimension ({tensor.shape[2]} vs {embedding_dim}) mismatch found in an input layer. Cannot blend/sum. Returning empty conditioning.")
                summary = "Shape mismatch in input layers. Cannot blend/sum. Returning empty conditioning."
                if input_errors: summary += f"\nInput Errors/Warnings ({len(input_errors)}):\n" + "\n".join(input_errors)
                print(f"TenosaiFluxConditioningSummedBlend (Node ID {unique_id}): {summary}")
                if unique_id is not None: self._send_feedback(unique_id, "summed_blend_summary", summary)
                return ([], summary)

        # Apply layer-wise input normalizations AFTER validation and BEFORE padding/aggregation
        if normalize_to_input_layers:
             cond_to_layers = [[normalize_tensor(t), d] for t, d in cond_to_layers]
        if normalize_from_input_layers:
             cond_from_layers = [[normalize_tensor(t), d] for t, d in cond_from_layers]


        # --- Aggregate, Blend per Block, Collect Results for Summing ---
        blended_block_tensors_for_summing = []
        output_summary_lines = [
            f"Cond1 layers: {len(cond_to_layers)}, Cond2 layers: {len(cond_from_layers)}",
            f"Max sequence length: {max_seq_len}",
            f"Embedding dimension: {embedding_dim}",
            f"Batch size: {batch_size}",
            "--- Block Blending & Summing Contributions ---"
        ]
        if normalize_to_input_layers: output_summary_lines[0] += " (Cond1 layers normalized)"
        if normalize_from_input_layers: output_summary_lines[0] += " (Cond2 layers normalized)"


        block_ratios = {
            'encoder': encoder_ratio,
            'base': base_ratio,
            'decoder': decoder_ratio,
            'out': out_ratio,
        }

        ordered_flux_blocks = ['encoder', 'base', 'decoder', 'out']

        processed_block_count = 0
        for block_name in ordered_flux_blocks:
            # Aggregate tensors for this block from both inputs
            agg_to_tensor = _aggregate_layers(cond_to_layers, block_name, max_seq_len, embedding_dim, device, dtype)
            agg_from_tensor = _aggregate_layers(cond_from_layers, block_name, max_seq_len, embedding_dim, device, dtype)

            # Get the blend strength value for this output block name
            blend_ratio = block_ratios.get(block_name, 0.0)

            effective_ratio = blend_ratio
            if clamp_ratios:
                 effective_ratio = max(0.0, min(1.0, blend_ratio))
                 if abs(effective_ratio - blend_ratio) > 1e-6:
                      output_summary_lines.append(f" - '{block_name}': Ratio {blend_ratio:.4f} clamped to {effective_ratio:.4f}.")

            # --- Perform Blending for this specific block's contribution ---
            # Formula: agg_to * (1.0 - effective_ratio) + agg_from * effective_ratio
            blended_block_tensor = agg_to_tensor * (1.0 - effective_ratio) + agg_from_tensor * effective_ratio

            # Add the blended tensor for this block to the list for summing
            blended_block_tensors_for_summing.append(blended_block_tensor)
            processed_block_count += 1

            # Report the ratio and shapes in the summary
            output_summary_lines.append(f" - '{block_name}': Blended with ratio {effective_ratio:.4f} (Cond2 contribution). Agg To Shape: {agg_to_tensor.shape}, Agg From Shape: {agg_from_tensor.shape}.")


        # --- Sum the Blended Block Tensors ---
        if not blended_block_tensors_for_summing:
             # This shouldn't happen if inputs were valid and ordered_flux_blocks is not empty
             summary = "Internal Error: No blended block tensors generated for summing. Returning empty conditioning."
             if input_errors: summary += f"\nInput Warnings ({len(input_errors)}):\n" + "\n".join(input_errors)
             print(f"TenosaiFluxConditioningSummedBlend (Node ID {unique_id}): {summary}")
             if unique_id is not None: self._send_feedback(unique_id, "summed_blend_summary", summary)
             return ([], summary)

        final_tensor = torch.stack(blended_block_tensors_for_summing).sum(dim=0)
        summing_description = f"Summed {processed_block_count} blended block contribution tensors."

        # Optional Normalization of the final summed tensor
        if normalize_output:
            final_tensor = normalize_tensor(final_tensor)
            summing_description += " (Normalized after summing)"

        # --- Prepare Output Conditioning (Single Item List) ---
        # Create the dictionary for the output item by copying the dictionary
        # from the first valid 'to' item.
        base_output_dict = cond_to_layers[0][1].copy() if cond_to_layers else {}
        output_conditioning = [[final_tensor, base_output_dict]] # Output is a single item list

        # --- Final Summary ---
        summary = "\n".join(output_summary_lines) + "\n" + summing_description
        summary += f"\nOutput tensor shape: {final_tensor.shape}"
        if input_errors: summary += f"\nInput Warnings ({len(input_errors)}):\n" + "\n".join(input_errors)


        # Print final summary to console
        print(f"TenosaiFluxConditioningSummedBlend (Node ID {unique_id}):\n{summary}")

        # Send feedback via PromptServer to the UI, targeting the output string widget
        if unique_id is not None:
             self._send_feedback(unique_id, "summed_blend_summary", summary)

        # Return the final combined conditioning list (single item) and the summary string
        return (output_conditioning, summary)

    def _send_feedback(self, unique_id, widget_name, message):
         """Helper to send feedback to the ComfyUI frontend using PromptServer."""
         node_id_str = str(unique_id)
         try:
              if hasattr(PromptServer.instance, 'send_sync'):
                   PromptServer.instance.send_sync("impact-node-feedback", {
                       "node_id": node_id_str,
                       "widget_name": widget_name,
                       "type": "TEXT",
                       "value": message
                   })
              else:
                   pass
         except Exception as e:
              print(f"TenosaiFluxConditioningSummedBlend (Node ID {node_id_str}): Failed to send feedback via PromptServer: {e}")

# --- Node 4: TenosaiFluxConditioningAddScaledSum (Adds scaled contributions per block, outputs single) ---
class TenosaiFluxConditioningAddScaledSum:
    """
    FLUX Dev ONLY: Scales CONDITIONING_FROM per block using individual weights (0.0-10.0+),
    aggregating matching layers (and untagged) for each block. Sums these scaled block
    contribution tensors and adds the sum to the aggregated CONDITIONING_TO tensor
    (aggregated across all layers, including untagged). Preserves dictionary
    data from the first item of CONDITIONING_TO. Outputs single CONDITIONING item.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "conditioning_to": ("CONDITIONING", {"display": "conditioning_to", "tooltip": "Base conditioning source (Cond1). All layers aggregated. Original dictionary data preserved from its first valid item."}),
                "conditioning_from": ("CONDITIONING", {"display": "conditioning_from", "tooltip": "Modifying conditioning source (Cond2). Aggregated and scaled per block contribution."}),

                # Weight controls for scaling Conditioning From per Flux diffusion block (0.0-10.0+ range).
                "encoder_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Multiplier for Conditioning From's aggregate tensor for 'encoder' block contribution."}),
                "base_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Multiplier for Conditioning From's aggregate tensor for 'base' block contribution."}),
                "decoder_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Multiplier for Conditioning From's aggregate tensor for 'decoder' block contribution."}),
                "out_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Multiplier for Conditioning From's aggregate tensor for 'out' block contribution."}),

                 "normalize_to_input_layers": ("BOOLEAN", {"default": False, "display": "boolean", "tooltip": "Normalize each individual tensor from 'conditioning_to' BEFORE aggregation."}),
                 "normalize_from_input_layers": ("BOOLEAN", {"default": False, "display": "boolean", "tooltip": "Normalize each individual tensor from 'conditioning_from' BEFORE aggregation per block."}),
                 "normalize_output": ("BOOLEAN", {"default": False, "display": "boolean", "tooltip": "Normalize the final added output tensor."}),
            },
            "optional": {},
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
        }
        return inputs

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("CONDITIONING", "add_scaled_sum_summary")
    FUNCTION = "process_conditioning"
    CATEGORY = "Tenos.ai/Conditioning"
    DESCRIPTION = "FLUX Dev ONLY: Adds scaled Conditioning From (per block aggregate contribution) to total aggregated Conditioning To, outputs single item."

    def process_conditioning(self, conditioning_to, conditioning_from,
                             encoder_weight, base_weight, decoder_weight, out_weight,
                             normalize_to_input_layers, normalize_from_input_layers, normalize_output,
                             unique_id=None, extra_pnginfo=None):

        # --- Input Collection and Validation ---
        cond_to_layers, errors_to = _validate_and_extract_layered_conditioning(conditioning_to, "conditioning_to")
        cond_from_layers, errors_from = _validate_and_extract_layered_conditioning(conditioning_from, "conditioning_from")
        input_errors = errors_to + errors_from

        if not cond_to_layers or not cond_from_layers:
             summary = "One or both required conditioning inputs are missing or invalid. Cannot add. Returning empty conditioning."
             if input_errors: summary += f"\nInput Errors/Warnings ({len(input_errors)}):\n" + "\n".join(input_errors)
             print(f"TenosaiFluxConditioningAddScaledSum (Node ID {unique_id}): {summary}")
             if unique_id is not None: self._send_feedback(unique_id, "add_scaled_sum_summary", summary)
             return ([], summary)

        # Determine shapes and device from valid inputs
        all_input_layers = cond_to_layers + cond_from_layers
        max_seq_len = max(layer[0].shape[1] for layer in all_input_layers)
        batch_size = all_input_layers[0][0].shape[0]
        embedding_dim = all_input_layers[0][0].shape[2]
        device = all_input_layers[0][0].device
        dtype = all_input_layers[0][0].dtype

        # Check if batch size and embedding dim match across ALL inputs
        for tensor, dictionary in all_input_layers:
            if tensor.shape[0] != batch_size or tensor.shape[2] != embedding_dim:
                input_errors.append(f"Batch size ({tensor.shape[0]} vs {batch_size}) or Embedding dimension ({tensor.shape[2]} vs {embedding_dim}) mismatch found in an input layer. Cannot add. Returning empty conditioning.")
                summary = "Shape mismatch in input layers. Cannot add. Returning empty conditioning."
                if input_errors: summary += f"\nInput Errors/Warnings ({len(input_errors)}):\n" + "\n".join(input_errors)
                print(f"TenosaiFluxConditioningAddScaledSum (Node ID {unique_id}): {summary}")
                if unique_id is not None: self._send_feedback(unique_id, "add_scaled_sum_summary", summary)
                return ([], summary)

        # Apply layer-wise input normalizations AFTER validation and BEFORE padding/aggregation
        if normalize_to_input_layers:
             cond_to_layers = [[normalize_tensor(t), d] for t, d in cond_to_layers]
        if normalize_from_input_layers:
             cond_from_layers = [[normalize_tensor(t), d] for t, d in cond_from_layers]


        # --- Aggregate Cond To (all layers) ---
        agg_to_total_tensor = _aggregate_all_layers(cond_to_layers, max_seq_len, embedding_dim, device, dtype)

        # --- Aggregate Cond From per Block, Scale, and Collect for Summing ---
        scaled_cond_from_block_contributions = []
        output_summary_lines = [
            f"Cond1 layers: {len(cond_to_layers)}, Cond2 layers: {len(cond_from_layers)}",
            f"Max sequence length: {max_seq_len}",
            f"Embedding dimension: {embedding_dim}",
            f"Batch size: {batch_size}",
            f"Aggregated Cond To Shape (all layers): {agg_to_total_tensor.shape}",
            "--- Block Scaling & Summing Contributions from Cond From ---"
        ]
        if normalize_to_input_layers: output_summary_lines[0] += " (Cond1 layers normalized)"
        if normalize_from_input_layers: output_summary_lines[0] += " (Cond2 layers normalized)"


        block_weights = {
            'encoder': encoder_weight,
            'base': base_weight,
            'decoder': decoder_weight,
            'out': out_weight,
        }

        ordered_flux_blocks = ['encoder', 'base', 'decoder', 'out']

        processed_block_count = 0
        for block_name in ordered_flux_blocks:
            # Aggregate tensors for this block from conditioning_from (matching layers + untagged)
            agg_from_block_tensor = _aggregate_layers(cond_from_layers, block_name, max_seq_len, embedding_dim, device, dtype)

            # Get the weight value for this block
            multiplier = block_weights.get(block_name, 0.0)

            # --- Perform Scaling for this specific block's contribution from Cond From ---
            # Scale the aggregate block tensor by the block's multiplier.
            scaled_block_contribution = agg_from_block_tensor * multiplier

            # Add the scaled tensor for this block's contribution to the list for summing
            scaled_cond_from_block_contributions.append(scaled_block_contribution)
            processed_block_count += 1

            # Report the multiplier and shapes in the summary
            output_summary_lines.append(f" - '{block_name}': Multiplier {multiplier:.4f}. Agg From Shape: {agg_from_block_tensor.shape}, Scaled Contribution Shape: {scaled_block_contribution.shape}.")


        # --- Sum the Scaled Block Contribution Tensors from Conditioning From ---
        if not scaled_cond_from_block_contributions:
             # This shouldn't happen if inputs were valid and ordered_flux_blocks is not empty
             summary = "Internal Error: No scaled block contribution tensors generated for summing. Returning empty conditioning."
             if input_errors: summary += f"\nInput Warnings ({len(input_errors)}):\n" + "\n".join(input_errors)
             print(f"TenosaiFluxConditioningAddScaledSum (Node ID {unique_id}): {summary}")
             if unique_id is not None: self._send_feedback(unique_id, "add_scaled_sum_summary", summary)
             return ([], summary)

        summed_scaled_from_contributions = torch.stack(scaled_cond_from_block_contributions).sum(dim=0)
        summing_description = f"Summed {processed_block_count} scaled block contribution tensors from Conditioning From."


        # --- Add the Sum to the Aggregated Conditioning To Tensor ---
        final_tensor = agg_to_total_tensor + summed_scaled_from_contributions
        adding_description = f"Added sum of scaled Conditioning From block contributions to total aggregated Conditioning To tensor. Resulting shape {final_tensor.shape}."


        # Optional Normalization of the final added tensor
        if normalize_output:
            final_tensor = normalize_tensor(final_tensor)
            adding_description += " (Normalized after adding)"

        # --- Prepare Output Conditioning (Single Item List) ---
        # Create the dictionary for the output item by copying the dictionary
        # from the first valid 'to' item.
        base_output_dict = cond_to_layers[0][1].copy() if cond_to_layers else {}
        output_conditioning = [[final_tensor, base_output_dict]] # Output is a single item list

        # --- Final Summary ---
        summary = "\n".join(output_summary_lines) + "\n" + summing_description + "\n" + adding_description
        summary += f"\nOutput tensor shape: {final_tensor.shape}"
        if input_errors: summary += f"\nInput Warnings ({len(input_errors)}):\n" + "\n".join(input_errors)


        # Print final summary to console
        print(f"TenosaiFluxConditioningAddScaledSum (Node ID {unique_id}):\n{summary}")

        # Send feedback via PromptServer to the UI, targeting the output string widget
        if unique_id is not None:
             self._send_feedback(unique_id, "add_scaled_sum_summary", summary)

        # Return the final combined conditioning list (single item) and the summary string
        return (output_conditioning, summary)

    def _send_feedback(self, unique_id, widget_name, message):
         """Helper to send feedback to the ComfyUI frontend using PromptServer."""
         node_id_str = str(unique_id)
         try:
              if hasattr(PromptServer.instance, 'send_sync'):
                   PromptServer.instance.send_sync("impact-node-feedback", {
                       "node_id": node_id_str,
                       "widget_name": widget_name,
                       "type": "TEXT",
                       "value": message
                   })
              else:
                   pass
         except Exception as e:
              print(f"TenosaiFluxConditioningAddScaledSum (Node ID {node_id_str}): Failed to send feedback via PromptServer: {e}")


# --- Node 5: TenosaiFluxConditioningSpatialWeightedConcat (Weighted spatial concat, outputs single) ---
class TenosaiFluxConditioningSpatialWeightedConcat:
    """
    FLUX Dev ONLY: Combines two CONDITIONING inputs (Cond1 = to, Cond2 = from)
    via spatial tensor concatenation. Aggregates all layers from each input
    separately. The sum of block weight controls acts as an overall multiplier
    (0.0-10.0+) for the *total aggregated* CONDITIONING_FROM tensor before concatenation.
    Preserves original conditioning data like 'pooled_output' from the first
    item of CONDITIONING_TO. Outputs single spatially concatenated CONDITIONING item.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "conditioning_to": ("CONDITIONING", {"display": "conditioning_to", "tooltip": "Base conditioning source (Cond1). All layers aggregated. Tensor will be the first part of concatenated tensor. Original dictionary data preserved from its first valid item."}),
                "conditioning_from": ("CONDITIONING", {"display": "conditioning_from", "tooltip": "Modifying conditioning source (Cond2). All layers aggregated, then scaled by the total weight sum and concatenated."}),

                # Weight controls (0.0-10.0+) that contribute to the overall multiplier for *total aggregated* conditioning_from.
                # Sum of these values is the multiplier.
                "encoder_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Weight contribution for 'encoder' block. Sums with others to form overall multiplier for total aggregated Cond2 tensor."}),
                "base_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Weight contribution for 'base' block. Sums with others to form overall multiplier for total aggregated Cond2 tensor."}),
                "decoder_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Weight contribution for 'decoder' block. Sums with others to form overall multiplier for total aggregated Cond2 tensor."}),
                "out_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "display": "number", "tooltip": "Weight contribution for 'out' block. Sums with others to form overall multiplier for total aggregated Cond2 tensor."}),

                "normalize_to_input_layers": ("BOOLEAN", {"default": False, "display": "boolean", "tooltip": "Normalize each individual tensor from 'conditioning_to' BEFORE aggregation."}),
                "normalize_from_input_layers": ("BOOLEAN", {"default": False, "display": "boolean", "tooltip": "Normalize each individual tensor from 'conditioning_from' BEFORE aggregation."}),
                "normalize_final_output": ("BOOLEAN", {"default": False, "display": "boolean", "tooltip": "Normalize the final concatenated tensor."}),
            },
            "optional": {},
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
        }
        return inputs

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("CONDITIONING", "spatial_weighted_concat_summary")
    FUNCTION = "process_conditioning"
    CATEGORY = "Tenos.ai/Conditioning"
    DESCRIPTION = "FLUX Dev ONLY: Spatially concatenates total aggregated Conditioning To and scaled total aggregated Conditioning From tensors (scaled by sum of block weights)."

    def process_conditioning(self, conditioning_to, conditioning_from,
                             encoder_weight, base_weight, decoder_weight, out_weight,
                             normalize_to_input_layers, normalize_from_input_layers, normalize_final_output,
                             unique_id=None, extra_pnginfo=None):

        # --- Input Collection and Validation ---
        cond_to_layers, errors_to = _validate_and_extract_layered_conditioning(conditioning_to, "conditioning_to")
        cond_from_layers, errors_from = _validate_and_extract_layered_conditioning(conditioning_from, "conditioning_from")
        input_errors = errors_to + errors_from

        if not cond_to_layers or not cond_from_layers:
             summary = "One or both required conditioning inputs are missing or invalid. Cannot concatenate. Returning empty conditioning."
             if input_errors: summary += f"\nInput Errors/Warnings ({len(input_errors)}):\n" + "\n".join(input_errors)
             print(f"TenosaiFluxConditioningSpatialWeightedConcat (Node ID {unique_id}): {summary}")
             if unique_id is not None: self._send_feedback(unique_id, "spatial_weighted_concat_summary", summary)
             return ([], summary)

        # Determine shapes and device from valid inputs
        all_input_layers = cond_to_layers + cond_from_layers
        max_seq_len = max(layer[0].shape[1] for layer in all_input_layers)
        batch_size = all_input_layers[0][0].shape[0]
        embedding_dim = all_input_layers[0][0].shape[2]
        device = all_input_layers[0][0].device
        dtype = all_input_layers[0][0].dtype

        # Check if batch size and embedding dim match across ALL inputs
        for tensor, dictionary in all_input_layers:
            if tensor.shape[0] != batch_size or tensor.shape[2] != embedding_dim:
                input_errors.append(f"Batch size ({tensor.shape[0]} vs {batch_size}) or Embedding dimension ({tensor.shape[2]} vs {embedding_dim}) mismatch found in an input layer. Cannot concatenate. Returning empty conditioning.")
                summary = "Shape mismatch in input layers. Cannot concatenate. Returning empty conditioning."
                if input_errors: summary += f"\nInput Errors/Warnings ({len(input_errors)}):\n" + "\n".join(input_errors)
                print(f"TenosaiFluxConditioningSpatialWeightedConcat (Node ID {unique_id}): {summary}")
                if unique_id is not None: self._send_feedback(unique_id, "spatial_weighted_concat_summary", summary)
                return ([], summary)


        # Apply layer-wise input normalizations AFTER validation and BEFORE padding/aggregation
        if normalize_to_input_layers:
             cond_to_layers = [[normalize_tensor(t), d] for t, d in cond_to_layers]
        if normalize_from_input_layers:
             cond_from_layers = [[normalize_tensor(t), d] for t, d in cond_from_layers]


        # --- Aggregate Cond To (all layers) ---
        agg_to_total_tensor = _aggregate_all_layers(cond_to_layers, max_seq_len, embedding_dim, device, dtype)

        # --- Aggregate Cond From (all layers) ---
        agg_from_total_tensor = _aggregate_all_layers(cond_from_layers, max_seq_len, embedding_dim, device, dtype)


        # --- Calculate Overall Multiplier for Aggregated Conditioning From ---
        total_multiplier = (
            encoder_weight +
            base_weight +
            decoder_weight +
            out_weight
        )
        multiplier_description = f"Sum of block weights = {total_multiplier:.4f} (overall multiplier for total aggregated Conditioning From)."


        # --- Apply Multiplier and Perform Spatial Concatenation ---
        scaled_agg_from_total_tensor = agg_from_total_tensor * total_multiplier

        # Spatially concatenate
        final_tensor = torch.cat((agg_to_total_tensor, scaled_agg_from_total_tensor), dim=1)
        concat_description = f"Spatially concatenated total aggregated conditioning_to tensor ({agg_to_total_tensor.shape[1]} seq len) with scaled total aggregated conditioning_from tensor ({scaled_agg_from_total_tensor.shape[1]} seq len)."
        concat_description += f"\nResulting shape: {final_tensor.shape}"


        # Optional Normalization of the final concatenated tensor
        if normalize_final_output:
            if final_tensor is not None:
                final_tensor = normalize_tensor(final_tensor)
                concat_description += " (Normalized after concatenation)"
            else:
                 input_errors.append("Normalization requested but final tensor is None.")


        # --- Prepare Output Conditioning (Single Item List) ---
        # Create the dictionary for the output item by copying the dictionary
        # from the first valid 'to' item.
        base_output_dict = cond_to_layers[0][1].copy() if cond_to_layers else {}
        output_conditioning = [[final_tensor, base_output_dict]] # Output is a single item list

        # --- Final Summary ---
        summary = multiplier_description + "\n" + concat_description
        if input_errors: summary += f"\nInput Warnings ({len(input_errors)}):\n" + "\n".join(input_errors)


        # Print final summary to console
        print(f"TenosaiFluxConditioningSpatialWeightedConcat (Node ID {unique_id}):\n{summary}")

        # Send feedback via PromptServer to the UI, targeting the output string widget
        if unique_id is not None:
             self._send_feedback(unique_id, "spatial_weighted_concat_summary", summary)

        # Return the final combined conditioning list (single item) and the summary string
        return (output_conditioning, summary)

    def _send_feedback(self, unique_id, widget_name, message):
         """Helper to send feedback to the ComfyUI frontend using PromptServer."""
         node_id_str = str(unique_id)
         try:
              if hasattr(PromptServer.instance, 'send_sync'):
                   PromptServer.instance.send_sync("impact-node-feedback", {
                       "node_id": node_id_str,
                       "widget_name": widget_name,
                       "type": "TEXT",
                       "value": message
                   })
              else:
                   pass
         except Exception as e:
              print(f"TenosaiFluxConditioningSpatialWeightedConcat (Node ID {node_id_str}): Failed to send feedback via PromptServer: {e}")

# --- NODE MAPPINGS ---
NODE_CLASS_MAPPINGS = {
    "TenosaiFluxConditioningApply": TenosaiFluxConditioningApply,
    "TenosaiFluxConditioningBlend": TenosaiFluxConditioningBlend,
    "TenosaiFluxConditioningSummedBlend": TenosaiFluxConditioningSummedBlend,
    "TenosaiFluxConditioningAddScaledSum": TenosaiFluxConditioningAddScaledSum,
    "TenosaiFluxConditioningSpatialWeightedConcat": TenosaiFluxConditioningSpatialWeightedConcat,
}

# --- NODE DISPLAY NAMES ---
NODE_DISPLAY_NAME_MAPPINGS = {
    "TenosaiFluxConditioningApply": "Tenos.ai Flux Apply (Layered Input/Output)",
    "TenosaiFluxConditioningBlend": "Tenos.ai Flux Blend (Layered Input/Output)",
    "TenosaiFluxConditioningSummedBlend": "Tenos.ai Flux Summed Blend (Layered Input, Single Output)",
    "TenosaiFluxConditioningAddScaledSum": "Tenos.ai Flux Add Scaled (Layered Input, Single Output)",
    "TenosaiFluxConditioningSpatialWeightedConcat": "Tenos.ai Flux Concat (Spatial Weighted, Layered Input)",
}
# --- END OF FILE tenosai_flux_conditioning.py ---