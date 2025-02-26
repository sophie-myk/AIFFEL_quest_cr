import 'dart:async';
import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());  // 앱 실행
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pomodoro Timer',
      theme: ThemeData(
        primarySwatch: Colors.blue, // 기본 색상 설정
        scaffoldBackgroundColor: Colors.blue.shade50, // 전체 배경색
        textTheme: const TextTheme(
          bodyLarge: TextStyle(fontSize: 30, fontWeight: FontWeight.bold),
          bodyMedium: TextStyle(fontSize: 50, fontWeight: FontWeight.bold),
        ),
      ),
      home: const PomodoroTimer(),
    );
  }
}

class PomodoroTimer extends StatefulWidget {
  const PomodoroTimer({super.key});

  @override
  _PomodoroTimerState createState() => _PomodoroTimerState();
}

class _PomodoroTimerState extends State<PomodoroTimer> {
  int timeLeft = 25;  // 남은 시간 (초)
  bool isWorkTime = true; // 작업 중인지 여부
  int cycle = 1;  // 현재 사이클 (1~4)
  Timer? timer;  // 타이머 객체

  // 타이머 시작
  void startTimer() {
    timer?.cancel(); // 기존 타이머 취소

    timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      setState(() {
        timeLeft--; // 시간 줄이기
      });

      print('남은 시간: $timeLeft초, 현재 상태: ${isWorkTime ? "작업" : "휴식"}');

      if (timeLeft <= 0) {
        nextCycle();  // 다음 사이클로 전환
      }
    });
  }

  // 다음 사이클로 전환
  void nextCycle() {
    timer?.cancel(); // 타이머 정지

    print('사이클 변경: 현재 사이클 $cycle, ${isWorkTime ? "작업 종료" : "휴식 종료"}');

    setState(() {
      if (isWorkTime) {
        isWorkTime = false;
        timeLeft = (cycle == 4) ? 15 : 5; // 4번째 사이클은 15초 휴식
      } else {
        isWorkTime = true;
        timeLeft = 25;  // 다시 작업 시간
        cycle = (cycle % 4) + 1; // 사이클 1~4 반복
      }
    });

    startTimer(); // 타이머 다시 시작
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Pomodoro Timer')), // 상단 제목
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              isWorkTime ? '작업 중' : '휴식 중',
              style: Theme.of(context).textTheme.bodyLarge,
            ),
            Text(
              '$timeLeft 초',
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            Text(
              '사이클: $cycle',
              style: Theme.of(context).textTheme.bodyLarge,
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: startTimer,  // 버튼을 누르면 타이머 시작
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blueAccent,
                padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
              ),
              child: const Text('시작', style: TextStyle(fontSize: 25, color: Colors.white)),
            ),
          ],
        ),
      ),
    );
  }
}

