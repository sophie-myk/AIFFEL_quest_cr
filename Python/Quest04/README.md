# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 민유경
- 리뷰어 : 권이삭

# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - 중요! 해당 조건을 만족하는 부분을 캡쳐해 근거로 첨부
        - ![화면 캡처 2025-01-15 122848](https://github.com/user-attachments/assets/5c6fbf5c-6452-43b1-9d38-8f7a4b928c80)

    
- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
  주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        - ![화면 캡처 2025-01-15 123500](https://github.com/user-attachments/assets/ee2042ed-492a-4d73-8a3b-dd29f800fca5)
        - 주석이 깔끔하게 정리되어 있었고 가장 중요한 예외처리 부분을 잘 실행시켰다고 생각합니다.

  
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
  새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도, 
      실험이 기록되어 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        - ![화면 캡처 2025-01-15 124937](https://github.com/user-attachments/assets/7cebb3d9-e216-48c3-afab-74827893d687)
        - ![화면 캡처 2025-01-15 124806](https://github.com/user-attachments/assets/f763f4fa-a005-48cd-9b80-494dd19ddd29)


        - 구체적으로 어디가 오류가 나며 어떻게 문제를 해결하려고 했는지 잘 정리되어 있었습니다.


  
- [x]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        - ![화면 캡처 2025-01-15 124516](https://github.com/user-attachments/assets/e981f7bd-3348-47e5-a302-c04be69c879d)
        - 회고 역시 정말 구체적으로 잘 작성하셨습니다.

    
- [X]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        - ![화면 캡처 2025-01-15 140944](https://github.com/user-attachments/assets/742e04b7-67e2-4864-8acb-8dfafebba793)
        - 요점을 잘 정리한 코드였다고 생각합니다. 주석도 잘 달렸습니다.



# 회고(참고 링크 및 코드 개선)
```
# 리뷰어의 회고를 작성합니다.
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.

처음에 몇가지의 오류가 있었지만 꼼꼼하게 체크를 하며 문제를 하나하나 잘 해결한 좋은 퀘스트였다고 생각합니다.
보통 정수에 대한 오류만 생각하는데 연산자의 오류까지 체크한 것이 굉장히 인상적이었습니다.
![화면 캡처 2025-01-15 141300](https://github.com/user-attachments/assets/3130d138-906c-4d82-b563-4a72a7ba0373)

계산을 다시 재실행하기 위한 코드가 y만 반응하는 것이 아닌 n의 경우도 만들고 y와 n이 아닌 경우에는 예외처리를 하는 것이 좋지 않았을까 생각합니다.
![화면 캡처 2025-01-15 141351](https://github.com/user-attachments/assets/d1ad804b-d85d-4577-b9b9-cc041be4ef0f)
while True:
            cont = input("계속 계산하시겠습니까? (y/n): ").strip().lower()
            if cont == 'y':
                break
            elif cont == 'n':
                print("계산기를 종료합니다.")
                return
            else:
                print("잘못된 버튼입니다. 'y' 또는 'n'을 입력해주세요.")

if __name__ == "__main__":
    main()

전체적으로 굉장히 완성도가 높은 코드였다고 생각합니다.

```
