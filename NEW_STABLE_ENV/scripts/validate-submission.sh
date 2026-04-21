#!/bin/bash
# scripts/validate-submission.sh
# Validates that the core files exist and are syntactically correct.

echo "🔍 Validating submission..."

REQUIRED_FILES=("models.py" "server/app.py" "server/environment.py" "gym_env/precision_irrigation.py")

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing file: $file"
        exit 1
    fi
done

echo "✅ All required files found."

# Run syntax check
echo "🔠 Checking syntax..."
python -m compileall -q .
if [ $? -eq 0 ]; then
    echo "✅ Syntax check passed."
else
    echo "❌ Syntax errors found."
    exit 1
fi

echo "🚀 Validation complete."
