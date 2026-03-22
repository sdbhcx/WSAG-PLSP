import re

log_file_path = 'train1.log'

try:
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
except FileNotFoundError:
    print(f"Error: File {log_file_path} not found.")
    exit()

data = []
current_epoch = -1

# Regex patterns
epoch_pattern = re.compile(r'============Training Epoch (\d+)============')
# Pattern for "New best KLD: KLD_val, SIM_val, NSS_val"
kld_pattern = re.compile(r'New best KLD:\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)')

headers = ['Epoch', 'KLD', 'SIM', 'NSS']
print('\t'.join(headers))

for line in lines:
    line = line.strip()
    
    # Check for epoch
    epoch_match = epoch_pattern.search(line)
    if epoch_match:
        current_epoch = int(epoch_match.group(1))
        continue
    
    # Check for New best KLD
    kld_match = kld_pattern.search(line)
    if kld_match:
        kld = kld_match.group(1)
        sim = kld_match.group(2)
        nss = kld_match.group(3)
        
        row = [str(current_epoch), kld, sim, nss]
        print('\t'.join(row))
