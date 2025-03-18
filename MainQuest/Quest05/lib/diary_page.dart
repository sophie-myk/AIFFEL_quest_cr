import 'dart:math';
import 'package:flutter/material.dart';

class DiaryPage extends StatefulWidget {
  final String imagePath; // 문자열 경로를 받도록 수정

  DiaryPage({required this.imagePath});

  @override
  _DiaryPageState createState() => _DiaryPageState();
}

class _DiaryPageState extends State<DiaryPage> {
  final TextEditingController _controller = TextEditingController();
  String _responseText = "";

  // 미리 저장된 답변 리스트
  final List<String> _responses = [
    "정말 힘들었겠구나. 그래도 오늘 참 잘 해냈어.",
    "걔는 무슨 말을 그렇게 한대니. 대신 욕해줄까?",
    "오늘은 그냥 쉬자. 달콤한 딸기케이크를 먹는게 도움이 될 것 같아.",
    "스트레스 폭망이다. 오늘은 마라탕으로 가자.",
    "우리 오늘 바람이나 쐬러 갈까?",
    "미안. 영어는 잘 이해못해.",
    "나 같아도 그런 일은 정말 어려웠을 것 같아. 넌 대단해!",
    "오늘은 이만 자는 게 어때? 충분한 수면이 도움이 된대."
  ];

  // 랜덤 답변 선택
  String _getRandomResponse() {
    final randomIndex = Random().nextInt(_responses.length);
    return _responses[randomIndex];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Diary with InsideOut'),
      ),
      body: Column(
        children: [
          Expanded(
            child: Center(
              child: Image.asset(
                widget.imagePath,
                fit: BoxFit.cover,
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: TextField(
              controller: _controller,
              decoration: InputDecoration(
                border: OutlineInputBorder(),
                labelText: '텍스트를 입력하세요',
              ),
              maxLines: 4,
              keyboardType: TextInputType.multiline,
              textInputAction: TextInputAction.newline,
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: ElevatedButton(
              onPressed: () {
                String randomResponse = _getRandomResponse();
                setState(() {
                  _responseText = randomResponse; // 랜덤 응답을 출력
                });
              },
              child: Text('제출'),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Text(
              _responseText,
              style: TextStyle(fontSize: 16),
            ),
          ),
        ],
      ),
    );
  }
}