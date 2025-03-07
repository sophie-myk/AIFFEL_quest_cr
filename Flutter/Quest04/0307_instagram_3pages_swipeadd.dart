import 'package:flutter/material.dart';
import 'package:cached_network_image/cached_network_image.dart';

// 앱 실행 진입점
void main() {
  runApp(const MyApp());
}

// 최상위 앱 위젯
class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: HomeScreen(), // 홈 화면을 첫 화면으로 설정
    );
  }
}

// 홈 화면을 관리하는 StatefulWidget
class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0; // 현재 선택된 네비게이션 인덱스

  // 네비게이션 바에서 사용할 화면 리스트
  final List<Widget> _screens = [
    InstagramHome(),
    Center(child: Text("Search Screen")), // 검색 화면 (추후 구현 가능)
    Center(child: Text("Upload Screen")), // 업로드 화면 (추후 구현 가능)
    Center(child: Text("Notifications Screen")), // 알림 화면 (추후 구현 가능)
    ProfileScreen(),
  ];

  // 네비게이션 바 클릭 시 상태 업데이트
  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _screens[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: ''),
          BottomNavigationBarItem(icon: Icon(Icons.search), label: ''),
          BottomNavigationBarItem(icon: Icon(Icons.add_box_outlined), label: ''),
          BottomNavigationBarItem(icon: Icon(Icons.favorite_border), label: ''),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: ''),
        ],
        type: BottomNavigationBarType.fixed,
      ),
    );
  }
}

// 인스타그램 홈 화면 UI
class InstagramHome extends StatelessWidget {
  final List<String> stories = List.generate(10, (index) => 'https://picsum.photos/200/300?random=$index');
  final List<String> posts = List.generate(5, (index) => 'https://picsum.photos/500/500?random=$index');

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        title: Text('Instagram', style: TextStyle(fontFamily: 'Billabong', fontSize: 32, color: Colors.black)),
        elevation: 0,
      ),
      body: Column(
        children: [
          // 스토리 목록 (가로 스크롤)
          SizedBox(
            height: 120,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              itemCount: stories.length,
              itemBuilder: (context, index) {
                return GestureDetector(
                  onTap: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => StoryScreen(storyUrl: stories[index])),
                    );
                  },
                  child: Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: CircleAvatar(
                      radius: 45,
                      backgroundImage: CachedNetworkImageProvider(stories[index]),
                    ),
                  ),
                );
              },
            ),
          ),
          // 게시물 목록 (세로 스크롤)
          Expanded(
            child: ListView.builder(
              itemCount: posts.length,
              itemBuilder: (context, index) {
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    ListTile(
                      leading: CircleAvatar(
                        backgroundImage: CachedNetworkImageProvider(stories[index % stories.length]),
                      ),
                      title: Text('User $index', style: TextStyle(fontWeight: FontWeight.bold)),
                    ),
                    CachedNetworkImage(imageUrl: posts[index]),
                    Row(
                      children: [
                        IconButton(icon: Icon(Icons.favorite_border), onPressed: () {}),
                        IconButton(icon: Icon(Icons.comment), onPressed: () {}),
                        IconButton(icon: Icon(Icons.send), onPressed: () {}),
                      ],
                    ),
                    Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Text('Liked by user123 and others', style: TextStyle(fontWeight: FontWeight.bold)),
                    ),
                  ],
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

// 프로필 화면 UI
class ProfileScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Profile'),
        centerTitle: true,
      ),
      body: Center(
        child: Text('This is the Profile Screen', style: TextStyle(fontSize: 20)),
      ),
    );
  }
}

// 스토리 화면 UI
class StoryScreen extends StatelessWidget {
  final String storyUrl;

  StoryScreen({required this.storyUrl});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: GestureDetector(
        onHorizontalDragEnd: (details) {
          if (details.velocity.pixelsPerSecond.dx > 0) {
            // 오른쪽으로 드래그했을 때
            Navigator.pop(context);
          }
        },
        child: Center(
          child: CachedNetworkImage(
            imageUrl: storyUrl,
            fit: BoxFit.contain,
          ),
        ),
      ),
    );
  }
}
