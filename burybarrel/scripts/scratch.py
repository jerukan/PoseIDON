import concurrent.futures

import torch


def test_process(device):
    t = torch.tensor([1, 2, 3], device=device)
    print(f"Testing process on device: {device}\ntorch available: {torch.cuda.is_available()}\ntensor: {t}\n")


devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"]
with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
    future_to_res = {
        executor.submit(test_process, device): device
        for device in devices
    }
    for future in concurrent.futures.as_completed(future_to_res):
        device = future_to_res[future]
        try:
            future.result()
        except Exception as e:
            print("xd")
