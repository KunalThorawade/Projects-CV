#include <SFML/Graphics.hpp>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>
#include <fstream> // For file existence check

using namespace std;
using namespace sf;

#define N 9
#define UNASSIGNED 0

bool fileExists(const string& filename) {
    ifstream file(filename);
    return file.good();
}

void drawGrid(RenderWindow& window, const vector<vector<int>>& board, Font& font) {
    RectangleShape line(Vector2f(540, 1));
    line.setFillColor(Color::Black);

    for (int i = 0; i <= N; i++) {
        line.setPosition(0, i * 60);
        window.draw(line);
        line.setRotation(90);
        line.setPosition(i * 60, 0);
        window.draw(line);
        line.setRotation(0);
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (board[i][j] != UNASSIGNED) {
                Text number(to_string(board[i][j]), font, 24);
                number.setFillColor(Color::Black);
                number.setPosition(j * 60 + 20, i * 60 + 10);
                window.draw(number);
            }
        }
    }
}

bool isSafe(vector<vector<int>>& board, int row, int col, int num) {
    for (int x = 0; x < N; x++) {
        if (board[row][x] == num || board[x][col] == num || 
            board[row - row % 3 + x / 3][col - col % 3 + x % 3] == num) {
            return false;
        }
    }
    return true;
}

bool solveSudoku(RenderWindow& window, vector<vector<int>>& board, int row, int col, Font& font) {
    if (row == N - 1 && col == N)
        return true;
    if (col == N) {
        row++;
        col = 0;
    }
    if (board[row][col] != UNASSIGNED)
        return solveSudoku(window, board, row, col + 1, font);
    
    for (int num = 1; num <= N; num++) {
        if (isSafe(board, row, col, num)) {
            board[row][col] = num;

            // Update window and sleep for visualization
            window.clear(Color::White);
            drawGrid(window, board, font);
            window.display();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            if (solveSudoku(window, board, row, col + 1, font))
                return true;
            board[row][col] = UNASSIGNED;

            // Update window and sleep for visualization
            window.clear(Color::White);
            drawGrid(window, board, font);
            window.display();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    return false;
}

int main() {
    RenderWindow window(VideoMode(540, 540), "Sudoku Solver");

    vector<vector<int>> board = {
        {5, 3, 0, 0, 7, 0, 0, 0, 0},
        {6, 0, 0, 1, 9, 5, 0, 0, 0},
        {0, 9, 8, 0, 0, 0, 0, 6, 0},
        {8, 0, 0, 0, 6, 0, 0, 0, 3},
        {4, 0, 0, 8, 0, 3, 0, 0, 1},
        {7, 0, 0, 0, 2, 0, 0, 0, 6},
        {0, 6, 0, 0, 0, 0, 2, 8, 0},
        {0, 0, 0, 4, 1, 9, 0, 0, 5},
        {0, 0, 0, 0, 8, 0, 0, 7, 9}
    };

    string fontFilename = "arial.ttf";
    Font font;
    if (!fileExists(fontFilename)) {
        cerr << "Font file " << fontFilename << " does not exist" << endl;
        return -1;
    }
    if (!font.loadFromFile(fontFilename)) {
        cerr << "Could not load font" << endl;
        return -1;
    }

    std::thread solverThread([&]() {
        solveSudoku(window, board, 0, 0, font);
    });

    while (window.isOpen()) {
        Event event;
        while (window.pollEvent(event)) {
            if (event.type == Event::Closed)
                window.close();
        }

        window.clear(Color::White);
        drawGrid(window, board, font);
        window.display();
    }

    solverThread.join();
    return 0;
}
