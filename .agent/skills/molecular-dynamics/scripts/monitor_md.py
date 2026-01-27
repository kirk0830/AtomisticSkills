import argparse
import time
import os
import sys
import numpy as np

def monitor_md(log_path, target_temp, check_interval=60, timeout=60):
    """
    Monitor an ASE MD log file for unphysical instabilities and temperature drift.
    """
    print(f"Starting advanced real-time monitoring of {log_path}...")
    print(f"Target Temperature: {target_temp}K")
    
    last_mtime = 0
    start_time = time.time()
    starting_temp = None
    temp_history = []
    
    while True:
        if not os.path.exists(log_path):
            if time.time() - start_time > timeout:
                print(f"Error: Log file {log_path} was not created within {timeout} seconds.")
                sys.exit(1)
            time.sleep(5)
            continue
            
        current_mtime = os.path.getmtime(log_path)
        
        if current_mtime > last_mtime:
            last_mtime = current_mtime
            
            try:
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) < 2:
                        time.sleep(10)
                        continue
                    
                    header = lines[0].split()
                    
                    # 1. Identify Starting Temperature from the first frame
                    if starting_temp is None:
                        first_line = lines[1].split()
                        if len(first_line) == len(header):
                            data_start = dict(zip(header, first_line))
                            temp_key = [k for k in data_start if 'Temp' in k]
                            if temp_key:
                                starting_temp = float(data_start[temp_key[0]])
                                print(f"Detected starting temperature: {starting_temp:.1f}K")

                    # 2. Extract Recent Data
                    # Filter out empty or incomplete lines
                    processed_lines = [l.split() for l in lines[1:] if len(l.split()) == len(header)]
                    if not processed_lines:
                        time.sleep(10)
                        continue
                        
                    last_line_data = dict(zip(header, processed_lines[-1]))
                    temp_key = [k for k in last_line_data if 'Temp' in k]
                    
                    if not temp_key:
                        time.sleep(10)
                        continue
                        
                    current_temp = float(last_line_data[temp_key[0]])
                    temp_history.append(current_temp)
                    if len(temp_history) > 10: # Keep a window of 10
                        temp_history.pop(0)

                    # --- CRITICAL STABILITY CHECKS ---

                    # A. Absolute Explosion Detection
                    if current_temp > 10000 or np.isnan(current_temp):
                        print("CRITICAL: MLIP_EXPLOSION_DETECTED")
                        print(f"Unphysical Temperature: {current_temp}K")
                        sys.stdout.flush()

                    # B. Fluctuation Check (Last 4 readings)
                    if len(temp_history) >= 4:
                        recent_std = np.std(temp_history[-4:])
                        # If fluctuations exceed a huge threshold (e.g. 500K std dev)
                        # or if they are proportionally massive relative to target T
                        if recent_std > 500 or recent_std > (target_temp * 0.5):
                            print("CRITICAL: EXCESSIVE_TEMPERATURE_FLUCTUATIONS")
                            print(f"Recent fluctuations (std dev): {recent_std:.1f}K")
                            sys.stdout.flush()

                    # C. Drift/Direction Checks
                    if starting_temp is not None:
                        # Case 1: Cooling (Start > Target)
                        if starting_temp > target_temp:
                            # It should be decreasing. If it drops much lower than target:
                            if current_temp < (target_temp - 100):
                                print("CRITICAL: TEMPERATURE_OVERSHOOT_COOLING")
                                print(f"Current T ({current_temp:.1f}K) dropped > 100K below target ({target_temp}K)")
                                sys.stdout.flush()
                        
                        # Case 2: Heating (Start < Target)
                        elif starting_temp < target_temp:
                            # It should be increasing. If it exceeds target by too much:
                            if current_temp > (target_temp + 100):
                                print("CRITICAL: TEMPERATURE_OVERSHOOT_HEATING")
                                print(f"Current T ({current_temp:.1f}K) is > 100K above target ({target_temp}K)")
                                sys.stdout.flush()

            except Exception as e:
                print(f"Monitoring error: {e}")
        
        if time.time() - last_mtime > timeout:
            print(f"Monitoring ended: No updates for {timeout} seconds.")
            break
            
        time.sleep(check_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Real-time MD stability monitor.")
    parser.add_argument("log_path", help="Path to ASE MD log file")
    parser.add_argument("--target_temp", type=float, required=True, help="Target MD temperature in K")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds (default: 60)")
    parser.add_argument("--timeout", type=int, default=600, help="Exit if no updates for this long (default: 600)")
    
    args = parser.parse_args()
    
    monitor_md(args.log_path, args.target_temp, check_interval=args.interval, timeout=args.timeout)
