"""
Master Test Runner for All 4 Model Combinations
Runs all tests sequentially and saves outputs
"""
import os
import subprocess
import sys
from datetime import datetime

project_root = r"c:\Users\hardi\OneDrive\Desktop\MedAIExplainableFractureDetection"
scripts_dir = os.path.join(project_root, "scripts")
outputs_dir = os.path.join(project_root, "outputs")

tests = [
    ("test_combo_1.py", "Combo_1_maxvit_hypercolumn_focal_yolo"),
    ("test_combo_2.py", "Combo_2_maxvit_hypercolumn_focal_radino"),
    ("test_combo_3.py", "Combo_3_maxvit_yolo_radino"),
    ("test_combo_4.py", "Combo_4_hypercolumn_focal_radino_yolo"),
]

print("="*80)
print("MASTER TEST RUNNER - ENSEMBLE STACKING TESTS")
print("="*80)
print(f"Total tests to run: {len(tests)}")
print(f"Start time: {datetime.now()}")
print("="*80)

results_summary = []

for i, (script_name, output_name) in enumerate(tests, 1):
    script_path = os.path.join(scripts_dir, script_name)
    output_file = os.path.join(outputs_dir, f"test_{output_name}_results.txt")
    
    print(f"\n[{i}/{len(tests)}] Running {script_name}...")
    print(f"    Output: {output_file}")
    
    try:
        # Run test and capture output
        with open(output_file, 'w') as f:
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=project_root,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=1200  # 20 minute timeout per test
            )
        
        if result.returncode == 0:
            print(f"    ✅ SUCCESS - Results saved to {output_file}")
            results_summary.append(f"✅ {output_name}: PASSED")
        else:
            print(f"    ⚠️ COMPLETED with exit code {result.returncode}")
            results_summary.append(f"⚠️ {output_name}: Exit code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print(f"    ❌ TIMEOUT - Test exceeded 20 minutes")
        results_summary.append(f"❌ {output_name}: TIMEOUT")
    except Exception as e:
        print(f"    ❌ ERROR: {str(e)}")
        results_summary.append(f"❌ {output_name}: ERROR - {str(e)}")

# Print Summary
print("\n" + "="*80)
print("TEST EXECUTION SUMMARY")
print("="*80)
for summary in results_summary:
    print(summary)
print("="*80)
print(f"End time: {datetime.now()}")
print("\n📁 All results saved to outputs/ directory:")
for script_name, output_name in tests:
    output_file = os.path.join(outputs_dir, f"test_{output_name}_results.txt")
    print(f"   - {output_file}")
print("="*80)
