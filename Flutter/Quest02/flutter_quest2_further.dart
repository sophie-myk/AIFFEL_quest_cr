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




# 회고(참고 링크 및 코드 개선)
```
              Stack(
                alignment: Alignment.center, // 중앙 정렬   <<======
                children: [
                  Container(width: 300, height: 300, color: Colors.red),
                  Container(width: 240, height: 240, color: Colors.orange),
                  Container(width: 180, height: 180, color: Colors.yellow),
                  Container(width: 120, height: 120, color: Colors.green),
                  Container(width: 60, height: 60, color: Colors.lightBlue),

                  으로 작성했을때는 Stack안에 중앙정렬을 넣은 탓에 원하는 모습이 구현되지 않았다.
                  Elevated Butten에 적용된 중앙정렬이 Stack까지 적용될 수 있도록 해당 행을 삭제하니
                  원하는 모습이 잘 구현되었다.
  
제시된 문제에는 스텍의 색이 나와있지 않았지만 무지개색으로 배치함으로써 좀 더 예쁘게 보이게 했다.
"Text"버튼을 누르면 DEBUG CONSOLE에 "버튼이 눌렸습니다." 라는 문장이 출력되는 부분이 
화면자체에 보이면 더 좋을 것 같아서,
             void _showMessage() {
                setState(() {
                  message = "버튼을 눌러줘서 고마워☆"; // 버튼 클릭 시 메시지 변경
                });
              }
                  부분을 class하위에 배치해 화면에서 메시지가 직접 출력되도록 하였다.
Flutter로 넘어간 이후에 파이썬이 그리워질만큼 너무 어렵고 정신없는 과목이라는 생각이 들었는데, 막상 휴대폰처럼 구현된 애뮬레이터에 
하나하나씩 구현해내는 작업이 꽤 흥미진진해서 좋았다. 싫어지는만큼 또 더 좋아지는 부분도 있었으면 좋겠다.
앱 상단바에 색을 넣는 작업이 의외로 어려웠는데 backgroundColor라는 것을 이용해서 순식간에 해결되는 것도 신기했다.
화면을 구성할때는 내가 이 도형이나 버튼을 어디에 어떤순서와 어떤 배율로 위치시킬것이냐 하는 것을 미리 구상하고 코드를 짜는 것도 중요할 것 같다. 
                  
```
