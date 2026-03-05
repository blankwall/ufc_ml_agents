#!/usr/bin/env python3
"""Validation script for feature changes"""
import json
import re
import ast
import sys

errors = []
warnings = []

# 1. JSON Syntax Validation
print("1. Validating JSON syntax...")
try:
    with open('schema/feature_schema.json') as f:
        schema = json.load(f)
    print("   ✓ feature_schema.json is valid JSON")
except json.JSONDecodeError as e:
    errors.append({
        "check": "json_syntax",
        "file": "schema/feature_schema.json",
        "message": f"Line {e.lineno}: {e.msg}",
        "severity": "critical"
    })
    print(f"   ✗ feature_schema.json has JSON error: {e}")

try:
    with open('schema/monotone_constraints.json') as f:
        constraints = json.load(f)
    print("   ✓ monotone_constraints.json is valid JSON")
except json.JSONDecodeError as e:
    errors.append({
        "check": "json_syntax",
        "file": "schema/monotone_constraints.json",
        "message": f"Line {e.lineno}: {e.msg}",
        "severity": "critical"
    })
    print(f"   ✗ monotone_constraints.json has JSON error: {e}")

# 2. Python Syntax Validation
print("\n2. Validating Python syntax...")
try:
    with open('features/feature_exclusions.py') as f:
        code = f.read()
    ast.parse(code)
    print("   ✓ feature_exclusions.py has valid Python syntax")
except SyntaxError as e:
    errors.append({
        "check": "python_syntax",
        "file": "features/feature_exclusions.py",
        "message": f"Line {e.lineno}: {e.msg}",
        "severity": "critical"
    })
    print(f"   ✗ feature_exclusions.py has syntax error: {e}")

# 3. Check for duplicates and naming
print("\n3. Checking schema consistency...")
features = schema.get('features', [])
duplicates = []
seen = set()
for feature in features:
    if feature in seen:
        duplicates.append(feature)
    seen.add(feature)

if duplicates:
    for dup in duplicates:
        errors.append({
            "check": "schema_consistency",
            "file": "schema/feature_schema.json",
            "message": f"Duplicate feature name: {dup}",
            "severity": "critical"
        })
    print(f"   ✗ Found {len(duplicates)} duplicate features")
else:
    print("   ✓ No duplicate features found")

# 4. Naming conventions
print("\n4. Checking feature naming conventions...")
invalid_names = []
python_keywords = {'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
                   'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
                   'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
                   'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
                   'try', 'while', 'with', 'yield'}

for feature in features:
    if not re.match(r'^[a-z][a-z0-9_]*$', feature):
        invalid_names.append(feature)
        errors.append({
            "check": "feature_naming",
            "file": "schema/feature_schema.json",
            "message": f"Feature '{feature}' doesn't follow snake_case convention",
            "severity": "warning"
        })
    if feature in python_keywords:
        invalid_names.append(feature)
        errors.append({
            "check": "feature_naming",
            "file": "schema/feature_schema.json",
            "message": f"Feature '{feature}' is a Python keyword",
            "severity": "critical"
        })

if invalid_names:
    print(f"   ✗ Found {len(invalid_names)} features with naming issues")
else:
    print("   ✓ All features follow naming conventions")

# 5. Monotone constraints consistency
print("\n5. Validating monotone constraints...")
constraint_features = set(constraints.get('constraints', {}).keys())
schema_features = set(features)

missing_in_schema = constraint_features - schema_features
if missing_in_schema:
    for feat in missing_in_schema:
        errors.append({
            "check": "constraints_consistency",
            "file": "schema/monotone_constraints.json",
            "message": f"Feature '{feat}' in constraints but not in feature_schema.json",
            "severity": "critical"
        })
    print(f"   ✗ Found {len(missing_in_schema)} features in constraints not in schema")
else:
    print("   ✓ All constraint features exist in schema")

invalid_constraint_values = []
for feat, value in constraints.get('constraints', {}).items():
    if value not in [-1, 0, 1]:
        invalid_constraint_values.append((feat, value))
        errors.append({
            "check": "constraints_consistency",
            "file": "schema/monotone_constraints.json",
            "message": f"Feature '{feat}' has invalid constraint value {value} (must be -1, 0, or 1)",
            "severity": "critical"
        })

if invalid_constraint_values:
    print(f"   ✗ Found {len(invalid_constraint_values)} invalid constraint values")
else:
    print("   ✓ All constraint values are valid (-1, 0, or 1)")

# 6. Check that excluded features exist in the codebase
print("\n6. Checking excluded features exist in codebase...")
excluded_base = [
    "stance_orthodox", "stance_southpaw", "stance_switch",
    "both_grapplers", "both_strikers", "both_finishers",
    "draws", "age_in_prime", "age_past_prime"
]

import os
feature_files = []
for root, dirs, files in os.walk('features'):
    for file in files:
        if file.endswith('.py') and file != '__init__.py':
            feature_files.append(os.path.join(root, file))

missing_implementations = []
for base_feat in excluded_base:
    found = False
    for feat_file in feature_files:
        try:
            with open(feat_file) as f:
                content = f.read()
            if base_feat in content:
                found = True
                break
        except:
            pass
    if not found:
        missing_implementations.append(base_feat)

if missing_implementations:
    for feat in missing_implementations:
        warnings.append({
            "check": "schema_consistency",
            "file": "features/",
            "message": f"Excluded feature '{feat}' not found in codebase",
            "severity": "warning"
        })
    print(f"   ⚠ {len(missing_implementations)} excluded features not found in code")
else:
    print("   ✓ All excluded features exist in codebase")

# Generate output
critical_errors = [e for e in errors if e['severity'] == 'critical']
output = {
    "status": "fail" if critical_errors else "pass",
    "errors": errors + warnings,
    "checks_performed": {
        "json_syntax": "pass" if not any(e['check'] == 'json_syntax' and e['severity'] == 'critical' for e in errors) else "fail",
        "python_syntax": "pass" if not any(e['check'] == 'python_syntax' for e in errors) else "fail",
        "schema_consistency": "pass" if not any(e['check'] == 'schema_consistency' and e['severity'] == 'critical' for e in errors) else "fail",
        "feature_naming": "pass" if not any(e['check'] == 'feature_naming' and e['severity'] == 'critical' for e in errors) else "fail",
        "constraints_consistency": "pass" if not any(e['check'] == 'constraints_consistency' and e['severity'] == 'critical' for e in errors) else "fail"
    },
    "can_proceed": len(critical_errors) == 0,
    "summary": f"Found {len(critical_errors)} critical errors and {len(warnings)} warnings"
}

# Write validation results
output_dir = '/Users/tylerbohan/code/ufc_ml_agents/agent_loop/agent_artifacts/20260129_205553/iter_2'
os.makedirs(output_dir, exist_ok=True)
with open(f'{output_dir}/validation.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*60}")
print(f"VALIDATION COMPLETE")
print(f"{'='*60}")
print(f"Status: {output['status'].upper()}")
print(f"Can proceed: {output['can_proceed']}")
print(f"Critical errors: {len(critical_errors)}")
print(f"Warnings: {len(warnings)}")
print(f"\nResults written to: {output_dir}/validation.json")

sys.exit(0 if output['can_proceed'] else 1)
