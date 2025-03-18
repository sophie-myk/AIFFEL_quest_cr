import 'package:flutter/material.dart'; // Flutter Material 패키지 가져오기
import 'package:get/get.dart'; // GetX 패키지 가져오기
import 'package:dio/dio.dart'; // Dio 패키지 가져오기
import 'dart:io'; // 플랫폼 관련 기능을 위해 가져오기

void main() {
  runApp(MyApp()); // 앱 실행
}

// 앱의 기본 구조를 정의하는 MyApp 클래스
class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return GetMaterialApp(
      title: 'Diary App', // 앱 제목
      theme: ThemeData(
        primarySwatch: Colors.blue, // 기본 색상
      ),
      home: HomePage(), // 홈 페이지로 이동
    );
  }
}

// 홈 페이지 클래스
class HomePage extends StatelessWidget {
  // 이미지 파일 경로 리스트
  final List<String> images = [
    'assets/images/photo1.jpg',
    'assets/images/photo2.jpg',
    'assets/images/photo3.jpg',
    'assets/images/photo4.jpg',
    'assets/images/photo5.jpg',
    'assets/images/photo6.jpg',
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('다이어리 이미지 선택'), // 앱 바 제목
      ),
      body: GridView.builder(
        gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 2, // 한 줄에 2개의 이미지
          childAspectRatio: 1, // 이미지 비율
        ),
        itemCount: images.length, // 이미지 개수
        itemBuilder: (context, index) {
          return GestureDetector(
            onTap: () {
              // 이미지를 클릭하면 DiaryPage로 이동
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => DiaryPage(imagePath: images[index]),
                ),
              );
            },
            child: Card(
              child: Image.asset(images[index], fit: BoxFit.cover), // 이미지 표시
            ),
          );
        },
      ),
    );
  }
}

// 일기 페이지 클래스
class DiaryPage extends StatefulWidget {
  final String imagePath; // 선택한 이미지 경로

  DiaryPage({required this.imagePath}); // 생성자

  @override
  _DiaryPageState createState() => _DiaryPageState(); // 상태 클래스 생성
}

// 일기 페이지의 상태 클래스
class _DiaryPageState extends State<DiaryPage> {
  final TextEditingController _controller = TextEditingController(); // 입력 컨트롤러
  final Dio dio = Dio(); // Dio 인스턴스 생성
  final List<Message> _messages = []; // 대화 내용 저장 리스트

  // 챗봇에 메시지를 전송하는 메서드
  Future<void> _sendToChatbot() async {
    final String userMessage = _controller.text.trim(); // 사용자 입력 가져오기

    // API 키와 URL 설정
    final String apiKey = 'AIzaSyChsWkWjYfeGYQC7ixkKh1xmzSEkCd935k'; // API 키
    final String apiUrl = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent'; // API URL

    if (userMessage.isEmpty) {
      return; // 입력이 없으면 종료
    }

    try {
      // 사용자 메시지를 리스트에 추가
      setState(() {
        _messages.add(Message(sender: 'user', text: userMessage)); // 사용자 메시지 추가
      });

      // API에 POST 요청
      final response = await dio.post(
        apiUrl,
        queryParameters: {'key': apiKey}, // API 키 전송
        options: Options(
          headers: {
            'Content-Type': 'application/json', // JSON 형식으로 전송
          },
        ),
        data: {
          'contents': [
            {
              'parts': [
                {'text': userMessage} // 사용자 입력된 텍스트
              ],
            },
          ],
        },
      );

      // API 응답 처리
      if (response.statusCode == 200) {
        final generatedText = response.data['candidates'][0]['content']['parts'][0]['text'];
        setState(() {
          _messages.add(Message(sender: 'bot', text: generatedText)); // 챗봇 응답 추가
          _controller.clear(); // 입력 필드 초기화
        });
      }
    } catch (e) {
      // 에러 발생 시 처리
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('일기 페이지'), // 페이지 제목
      ),
      body: Column(
        children: [
          Image.asset(widget.imagePath, height: 200), // 선택한 이미지 표시
          Expanded(
            child: ListView.builder(
              itemCount: _messages.length, // 메시지 개수
              itemBuilder: (context, index) {
                return _buildMessageBubble(_messages[index]); // 말풍선 생성
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              children: [
                TextField(
                  controller: _controller, // 입력 필드에 컨트롤러 연결
                  decoration: InputDecoration(
                    labelText: '오늘의 일기', // 입력 필드 레이블
                    border: OutlineInputBorder(),
                  ),
                  onSubmitted: (value) {
                    _sendToChatbot(); // Enter 키로 전송
                  },
                ),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: _sendToChatbot, // 버튼 클릭 시 전송
                  child: Text('전송'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // 메시지를 말풍선 형태로 표시하는 메서드
  Widget _buildMessageBubble(Message message) {
    final isUserMessage = message.sender == 'user'; // 사용자 메시지 여부 확인
    return Align(
      alignment: isUserMessage ? Alignment.centerRight : Alignment.centerLeft, // 정렬
      child: Container(
        margin: EdgeInsets.symmetric(vertical: 5, horizontal: 10), // 여백 설정
        padding: EdgeInsets.all(10), // 패딩 설정
        decoration: BoxDecoration(
          color: isUserMessage ? Colors.blueAccent : Colors.grey[300], // 배경색 설정
          borderRadius: BorderRadius.only(
            topLeft: Radius.circular(10),
            topRight: Radius.circular(10),
            bottomLeft: isUserMessage ? Radius.circular(10) : Radius.zero,
            bottomRight: isUserMessage ? Radius.zero : Radius.circular(10),
          ),
        ),
        child: Text(
          message.text,
          style: TextStyle(color: isUserMessage ? Colors.white : Colors.black), // 텍스트 색상 설정
        ),
      ),
    );
  }
}

// 메시지 클래스 정의
class Message {
  final String sender; // 발신자
  final String text; // 메시지 텍스트

  Message({required this.sender, required this.text}); // 생성자
}