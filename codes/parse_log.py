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
loss_pattern_1 = re.compile(r'Training loss:\s*([\d\.]+),\s*KL loss:\s*([\d\.]+),\s*Sim loss:\s*([\d\.]+),\s*Exo CLS loss:\s*([\d\.]+),')
loss_pattern_2 = re.compile(r'Noun sim loss:\s*([\d\.]+),\s*Part sim loss:\s*([\d\.]+),\s*Proto loss:\s*([\d\.]+)')

current_loss_data = {}

for i, line in enumerate(lines):
    line = line.strip()
    
    # Check for epoch
    epoch_match = epoch_pattern.search(line)
    if epoch_match:
        current_epoch = int(epoch_match.group(1))
        continue
    
    # Check for first part of loss
    loss_match_1 = loss_pattern_1.search(line)
    if loss_match_1:
        current_loss_data = {
            'Epoch': current_epoch,
            'Training loss': loss_match_1.group(1),
            'KL loss': loss_match_1.group(2),
            'Sim loss': loss_match_1.group(3),
            'Exo CLS loss': loss_match_1.group(4)
        }
        
        # Look ahead for the second part (it might be on the next line)
        if i + 1 < len(lines):
            next_line = lines[i+1].strip()
            loss_match_2 = loss_pattern_2.search(next_line)
            if loss_match_2:
                current_loss_data['Noun sim loss'] = loss_match_2.group(1)
                current_loss_data['Part sim loss'] = loss_match_2.group(2)
                current_loss_data['Proto loss'] = loss_match_2.group(3)
                data.append(current_loss_data)
                current_loss_data = {}

# Print headers for Excel copy-paste (Tab separated)
headers = ['Epoch', 'Training loss', 'KL loss', 'Sim loss', 'Exo CLS loss', 'Noun sim loss', 'Part sim loss', 'Proto loss']
print('\t'.join(headers))

for row in data:
    values = [str(row.get(h, '')) for h in headers]
    print('\t'.join(values))
