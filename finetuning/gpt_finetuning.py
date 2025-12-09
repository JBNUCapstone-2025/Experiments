"""
GPT-4o-mini Fine-tuning Script
OpenAI API를 이용한 GPT-4o-mini 파인튜닝 예제입니다.
"""

import os
import json
import time
from openai import OpenAI
from typing import List, Dict

class GPTFineTuner:
    def __init__(self, api_key: str = None):
        """
        GPT 파인튜너 객체 초기화

        Args:
            api_key: OpenAI API 키 (None이면 환경 변수에서 자동으로 읽음)
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.file_id = None
        self.job_id = None

    def prepare_training_data(self, data: List[Dict], output_file: str = "training_data.jsonl"):
        """
        파인튜닝 훈련 데이터를 JSONL 파일로 저장합니다.

        Args:
            data: 학습 데이터 리스트 (각 항목은 {"messages": [...]} 형태)
            output_file: 저장할 JSONL 파일 경로

        Example:
            data = [
                {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                        {"role": "assistant", "content": "Hi! How can I help you today?"}
                    ]
                }
            ]
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"✔ 학습 데이터가 {output_file} 파일로 저장되었습니다. (총 {len(data)}개 항목)")
        return output_file

    def upload_training_file(self, file_path: str):
        """
        학습 데이터를 OpenAI에 업로드합니다.

        Args:
            file_path: JSONL 파일 경로
        """
        print(f"📤 파일 업로드 시작: {file_path}")

        with open(file_path, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )

        self.file_id = response.id
        print(f"✔ 파일 업로드 완료. 파일 ID: {self.file_id}")

        # 파일 처리 상태 확인
        while True:
            file_info = self.client.files.retrieve(self.file_id)
            if file_info.status == 'processed':
                print("✔ 파일 처리 완료")
                break
            elif file_info.status == 'error':
                raise Exception(f"파일 처리 중 오류 발생: {file_info.status_details}")

            print("⏳ 파일 처리 중...")
            time.sleep(2)

        return self.file_id

    def create_fine_tune_job(
        self,
        model: str = "gpt-4o-mini-2024-07-18",
        n_epochs: int = 3,
        batch_size: int = None,
        learning_rate_multiplier: float = None,
        suffix: str = None
    ):
        """
        파인튜닝 작업 생성

        Args:
            model: 파인튜닝할 모델 이름 (기본: gpt-4o-mini-2024-07-18)
            n_epochs: 학습 반복 횟수
            batch_size: 배치 크기 (선택)
            learning_rate_multiplier: 러닝레이트 배수 (선택)
            suffix: 최종 생성될 모델 이름 뒤에 붙는 사용자 정의 텍스트
        """
        if not self.file_id:
            raise ValueError("upload_training_file()을 먼저 실행하여 학습 파일을 업로드해야 합니다.")

        print(f"\n🚀 파인튜닝 작업 생성 중...")
        print(f"  - 모델: {model}")
        print(f"  - 에포크: {n_epochs}")

        hyperparameters = {"n_epochs": n_epochs}
        if batch_size:
            hyperparameters["batch_size"] = batch_size
        if learning_rate_multiplier:
            hyperparameters["learning_rate_multiplier"] = learning_rate_multiplier

        job_params = {
            "training_file": self.file_id,
            "model": model,
            "hyperparameters": hyperparameters
        }

        if suffix:
            job_params["suffix"] = suffix

        response = self.client.fine_tuning.jobs.create(**job_params)

        self.job_id = response.id
        print(f"✔ 파인튜닝 작업 생성 완료 (Job ID: {self.job_id})")

        return self.job_id

    def monitor_job(self, job_id: str = None, check_interval: int = 60):
        """
        파인튜닝 작업 상태 모니터링

        Args:
            job_id: 모니터링할 작업 ID
            check_interval: 상태 체크 간격 (초)
        """
        job_id = job_id or self.job_id
        if not job_id:
            raise ValueError("job_id가 없습니다. create_fine_tune_job()을 먼저 실행해주세요.")

        print(f"\n📡 파인튜닝 작업 모니터링 시작: {job_id}")
        print(f"⏱ 체크 간격: {check_interval}초\n")

        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)

            status = job.status
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 상태: {status}")

            if status == 'succeeded':
                print("\n🎉 파인튜닝 완료!")
                print(f"   ✔ 최종 모델: {job.fine_tuned_model}")
                return job.fine_tuned_model

            elif status == 'failed':
                print("\n❌ 파인튜닝 실패")
                if job.error:
                    print(f"  오류: {job.error}")
                raise Exception("Fine-tuning job failed")

            elif status == 'cancelled':
                print("\n⚠ 작업이 취소되었습니다.")
                return None

            time.sleep(check_interval)

    def list_jobs(self, limit: int = 10):
        """
        파인튜닝 작업 목록 조회

        Args:
            limit: 최대 조회 개수
        """
        jobs = self.client.fine_tuning.jobs.list(limit=limit)

        print(f"\n📄 파인튜닝 작업 목록 (최대 {limit}개):")
        print("-" * 80)

        for job in jobs.data:
            print(f"ID: {job.id}")
            print(f"  상태: {job.status}")
            print(f"  모델: {job.model}")
            if job.fine_tuned_model:
                print(f"  결과 모델: {job.fine_tuned_model}")
            print(f"  생성일: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.created_at))}")
            print("-" * 80)

        return jobs.data

    def cancel_job(self, job_id: str = None):
        """
        진행 중인 파인튜닝 작업 취소

        Args:
            job_id: 작업 ID
        """
        job_id = job_id or self.job_id
        if not job_id:
            raise ValueError("job_id가 없습니다. create_fine_tune_job()을 먼저 실행해주세요.")

        response = self.client.fine_tuning.jobs.cancel(job_id)
        print(f"✔ 작업 취소 완료: {job_id}")
        return response

    def test_model(self, model_name: str, test_messages: List[Dict]):
        """
        파인튜닝된 모델 테스트

        Args:
            model_name: 모델 이름
            test_messages: 테스트 메시지
        """
        print(f"\n🧪 모델 테스트: {model_name}")
        print("-" * 80)

        response = self.client.chat.completions.create(
            model=model_name,
            messages=test_messages
        )

        result = response.choices[0].message.content
        print(f"📝 출력: {result}")
        print("-" * 80)

        return result


def main():
    """
    메인 실행 흐름
    """
    # 1. 파인튜너 객체 생성
    fine_tuner = GPTFineTuner()

    # 2. emotion_train.jsonl 파일 경로 설정
    file_path = "emotion_train.jsonl"

    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} 파일을 찾을 수 없습니다.")

    # 3. 업로드
    fine_tuner.upload_training_file(file_path)

    # 5. 파인튜닝 작업 생성
    fine_tuner.create_fine_tune_job(
        model="gpt-4o-mini-2024-07-18",
        n_epochs=3,
        suffix="emotion"
    )

    # 6. 작업 상태 모니터링
    fine_tuned_model = fine_tuner.monitor_job(check_interval=30)

    # 7. 모델 테스트
    if fine_tuned_model:
        test_messages = [
            {"role": "user", "content": "i feel so happy today"}
        ]
        fine_tuner.test_model(fine_tuned_model, test_messages)


if __name__ == "__main__":
    # API 키는 환경 변수 또는 생성자 인자로 설정 가능
    main()
