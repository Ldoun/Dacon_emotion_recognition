import math
import torch
import tempfile

def train_step(device, loader, model, loss_fn):
    model.train()
    x,y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    
    output = model(x)
    
    loss = loss_fn(output, y)
    loss.backward()


def max_gpu_batch_size(loader_class, logger, model, loss_fn, max_batch_size=1024):
    device = torch.device(device)  # type: ignore
    device_max_mem = torch.cuda.get_device_properties(device.index).total_memory

    def test_run(batch_size):
        logger.debug(f"Trying a run with batch size {batch_size}")
        with tempfile.TemporaryDirectory() as temp_dir:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            loader = loader_class(batch_size=batch_size)
            try:
                train_step(device, loader, model, loss_fn)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.debug("Exceeded memory capacity")
                    return None
                else:
                    raise e
        usage = torch.cuda.max_memory_allocated(device)
        logger.debug(f"Registered usage: {usage} / {device_max_mem} B")
        return usage

    usage_with_min_size = 0
    # Find a majoration of max batch size as a power of two
    for exponent in range(math.floor(math.log2(max_batch_size)) + 1):
        max_size = 2 ** exponent
        usage_with_max_size = test_run(max_size)
        if usage_with_max_size is None:
            break
        # This will only change as long as we don't break out, at which point it will
        # equal the usage for the previous test run
        usage_with_min_size = usage_with_max_size
    if usage_with_max_size is not None:
        logger.warning(
            f"Ran out of examples without finding a match batch size (max tried: {max_size})"
            ", you probably want to try with more examples"
        )

    return usage_with_min_size
