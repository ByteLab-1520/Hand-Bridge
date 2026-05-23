#!/usr/bin/env python3
"""
학습 데이터 및 모델 삭제 유틸리티

사용법:
    python delete.py              # 학습 데이터만 삭제
    python delete.py --all        # 학습 데이터 + 모델 모두 삭제
    python delete.py --model-only # 모델만 삭제
"""

import os
import sys
import shutil
from pathlib import Path


def delete_training_data():
    """학습 데이터 폴더(data/) 삭제"""
    data_dir = Path(__file__).parent / "data"
    
    if data_dir.exists():
        print(f"📁 '{data_dir}' 폴더 삭제 중...")
        shutil.rmtree(data_dir)
        print("✅ 학습 데이터 삭제 완료")
    else:
        print("⚠️  data/ 폴더가 없습니다.")


def delete_model():
    """학습된 모델 파일 삭제"""
    models_dir = Path(__file__).parent / "models"
    
    files_to_delete = [
        models_dir / "sign_model.keras",
        models_dir / "labels.json",
        models_dir / "training_history.png",
    ]
    
    deleted_count = 0
    for file in files_to_delete:
        if file.exists():
            print(f"🗑️  '{file.name}' 삭제 중...")
            file.unlink()
            deleted_count += 1
    
    if deleted_count > 0:
        print(f"✅ 모델 파일 {deleted_count}개 삭제 완료")
    else:
        print("⚠️  삭제할 모델 파일이 없습니다.")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        # 학습 데이터 + 모델 모두 삭제
        print("\n🚨 모든 학습 데이터와 모델을 삭제합니다...")
        response = input("정말 삭제하시겠습니까? (yes/no): ").strip().lower()
        
        if response == "yes":
            delete_training_data()
            delete_model()
            print("\n✅ 모든 학습 데이터와 모델이 삭제되었습니다!")
        else:
            print("취소되었습니다.")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--model-only":
        # 모델만 삭제
        print("\n🚨 학습된 모델을 삭제합니다...")
        response = input("정말 삭제하시겠습니까? (yes/no): ").strip().lower()
        
        if response == "yes":
            delete_model()
            print("\n✅ 모델이 삭제되었습니다!")
        else:
            print("취소되었습니다.")
    
    else:
        # 기본값: 학습 데이터만 삭제
        print("\n🚨 학습 데이터를 삭제합니다...")
        response = input("정말 삭제하시겠습니까? (yes/no): ").strip().lower()
        
        if response == "yes":
            delete_training_data()
            print("\n✅ 학습 데이터가 삭제되었습니다!")
            print("💡 새로운 수어 단어를 학습할 준비가 되었습니다.")
        else:
            print("취소되었습니다.")


if __name__ == "__main__":
    main()
