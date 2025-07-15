#!/bin/bash

echo "▶️ SwiftSyntax 실행 중..."
cd swift_syntax_runner
swift build
.build/debug/swift_syntax_runner
cd ..

echo "▶️ Python 분석 중..."
python run_pipeline.py
