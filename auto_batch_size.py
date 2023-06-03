#Compute maximum Batch Size iteratively

import math
import torch
import tempfile

def train_step(device, max_length_file, bs, processor, model, loss_fn): #test if current batch size occur cuda oom
    model.train()
    x = [max_length_file] * bs
    x = torch.tensor([processor(x_t) for x_t in x], dtype=torch.float) #this will make n tensor from max length file of train set
    x, y = x.to(device), torch.zeros(bs, dtype=torch.long).to(device)
    
    output = model(x)
    
    loss = loss_fn(output, y)
    loss.backward()

def max_gpu_batch_size(device, processor, logger, model, loss_fn, max_length_file, max_batch_size=1024): #based on https://github.com/Lightning-AI/lightning/issues/1615#issuecomment-619940827
    def test_run(bs):
        logger.info(f"Trying a run with batch size {bs}")
        with tempfile.TemporaryDirectory() as temp_dir:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            try:
                train_step(device, max_length_file, bs, processor, model, loss_fn)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.info("Exceeded memory capacity")
                    return None
                else:
                    raise e
        return bs

    min_batch_size = 0
    # Find a majoration of max batch size as a power of two
    for exponent in range(math.floor(math.log2(max_batch_size)) + 1):
        max_size = 2 ** exponent
        batch_size = test_run(max_size)
        if batch_size is None:
            break
        # This will only change as long as we don't break out, at which point it will
        # equal the usage for the previous test run
        min_batch_size = batch_size
    if batch_size is not None:
        logger.info(
            f"Ran out of examples without finding a match batch size (max tried: {max_size})"
            ", you probably want to try with more examples"
        )
    logger.info(f"using {min_batch_size} for batch_size")

    return min_batch_size
