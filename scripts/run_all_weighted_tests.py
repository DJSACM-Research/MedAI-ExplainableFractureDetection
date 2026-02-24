"""
Master Script: Run all weighted ensemble tests and compare results
"""
import subprocess
import os
import json
from datetime import datetime

test_configs = [
    {
        "script": "test_weighted_4models.py",
        "name": "4-Model Ensemble (maxvit, hypercolumn_focal, yolo, rad_dino)",
        "models": 4
    },
    {
        "script": "test_weighted_combo1.py",
        "name": "Combo 1 (maxvit, hypercolumn_focal, yolo)",
        "models": 3
    },
    {
        "script": "test_weighted_combo2.py",
        "name": "Combo 2 (maxvit, hypercolumn_focal, rad_dino)",
        "models": 3
    },
    {
        "script": "test_weighted_combo3.py",
        "name": "Combo 3 (maxvit, yolo, rad_dino)",
        "models": 3
    },
    {
        "script": "test_weighted_combo4.py",
        "name": "Combo 4 (hypercolumn_focal, rad_dino, yolo)",
        "models": 3
    }
]

project_root = r"c:\Users\hardi\OneDrive\Desktop\MedAIExplainableFractureDetection"
scripts_dir = os.path.join(project_root, "scripts")
outputs_dir = os.path.join(project_root, "outputs")

print("=" * 80)
print("WEIGHTED ENSEMBLE - COMPREHENSIVE TEST SUITE")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

results = {}

for i, config in enumerate(test_configs, 1):
    script_name = config["script"]
    test_name = config["name"]
    
    print(f"\n[{i}/{len(test_configs)}] Running: {test_name}")
    print("-" * 80)
    
    script_path = os.path.join(scripts_dir, script_name)
    
    try:
        # Run test
        result = subprocess.run(
            ["python", script_path],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=2400
        )
        
        if result.returncode == 0:
            print(result.stdout)
            results[test_name] = {"status": "✅ PASSED"}
        else:
            print(f"⚠️ Test failed with exit code {result.returncode}")
            print(result.stderr)
            results[test_name] = {"status": "❌ FAILED"}
    except subprocess.TimeoutExpired:
        print(f"⚠️ Test timed out")
        results[test_name] = {"status": "⏱️ TIMEOUT"}
    except Exception as e:
        print(f"❌ Error running test: {e}")
        results[test_name] = {"status": "❌ ERROR"}

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY - WEIGHTED ENSEMBLE EVALUATION")
print("=" * 80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

for test_name, result in results.items():
    status = result.get("status", "UNKNOWN")
    print(f"{status} - {test_name}")

print()
print("=" * 80)
print("All weighted ensemble tests completed!")
print(f"Results saved to: {outputs_dir}")
print("=" * 80)
