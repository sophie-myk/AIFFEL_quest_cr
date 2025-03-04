import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false, // 디버그 배너 제거
      home: Scaffold(
        appBar: AppBar(
          backgroundColor: Colors.blue,
          title: Text('플러터 앱 만들기'),
          leading: IconButton(
            icon: Icon(Icons.menu), // 올바른 아이콘 사용
            onPressed: () {
              print("메뉴 클릭");
            },
          ),
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton(
                onPressed: () {
                  print("버튼이 눌렸습니다.");
                },
                child: Text("Text"),
              ),

              SizedBox(height: 50), // 버튼과 Stack 사이 간격 추가

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
        ),
      ),
    );
  }
}

