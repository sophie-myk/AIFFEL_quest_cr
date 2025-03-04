import 'package:flutter/material.dart';

void main() {
  runApp(MyApp()); // 앱 실행
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false, // 디버그 배너 제거
      home: Scaffold(
        appBar: AppBar(
          backgroundColor: Colors.blue, // 앱바 색상
          title: Text('플러터 앱 만들기'), // 앱바 제목
          leading: IconButton(
            icon: Icon(Icons.menu), // 왼쪽 상단 아이콘
            onPressed: () {
              print("메뉴 클릭"); // 아이콘 클릭 시 콘솔 출력
            },
          ),
        ),
        body: BodyWidget(), // 본문 UI
      ),
    );
  }
}

class BodyWidget extends StatefulWidget {
  @override
  _BodyWidgetState createState() => _BodyWidgetState();
}

class _BodyWidgetState extends State<BodyWidget> {
  String message = ""; // 화면에 표시할 메시지

  void _showMessage() {
    setState(() {
      message = "버튼을 눌러줘서 고마워☆"; // 버튼 클릭 시 메시지 변경
    });
  }

  @override
  Widget build(BuildContext context) {
    return Center( // 화면 중앙 정렬
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center, // 세로 중앙 정렬
        children: [
          ElevatedButton(
            onPressed: _showMessage, // 버튼 클릭 시 메시지 변경
            child: Text("Text"), // 버튼 내부 텍스트
          ),

          SizedBox(height: 20), // 버튼과 메시지 간격

          Text(
            message, // 화면에 메시지 출력
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.black),
          ),

          SizedBox(height: 50), // 메시지와 Stack 사이 간격

          Stack(
            children: [
              Container(width: 300, height: 300, color: Colors.red),
              Container(width: 240, height: 240, color: Colors.orange),
              Container(width: 180, height: 180, color: Colors.yellow),
              Container(width: 120, height: 120, color: Colors.green),
              Container(width: 60, height: 60, color: Colors.lightBlue),
            ],
          ),
        ],
      ),
    );
  }
}
