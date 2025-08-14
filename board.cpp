#include <SFML/Graphics.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace sf;

#define N 9

bool isSafe(vector<vector<int>>& board, int row, int col, int num) {
    for (int x = 0; x < N; x++) {
        if (board[row][x] == num || board[x][col] == num ||
            board[row - row % 3 + x / 3][col - col % 3 + x % 3] == num) {
            return false;
        }
    }
    return true;
}

bool solveSudoku(vector<vector<int>>& board, int row, int col, vector<vector<bool>>& fixedCells) {
    if (row == N - 1 && col == N)
        return true;
    if (col == N) {
        row++;
        col = 0;
    }
    if (fixedCells[row][col])
        return solveSudoku(board, row, col + 1, fixedCells);

    for (int num = 1; num <= N; num++) {
        if (isSafe(board, row, col, num)) {
            board[row][col] = num;
            if (solveSudoku(board, row, col + 1, fixedCells))
                return true;
            board[row][col] = 0;
        }
    }
    return false;
}

void drawGrid(RenderWindow& window, const vector<vector<int>>& board, const vector<vector<bool>>& fixedCells, Font& font) {
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
            if (board[i][j] != 0) {
                Text number(to_string(board[i][j]), font, 24);
                number.setPosition(j * 60 + 20, i * 60 + 10);
                if (fixedCells[i][j]) {
                    number.setFillColor(Color::Black);
                } else {
                    number.setFillColor(Color::Red);
                }
                window.draw(number);
            }
        }
    }
}

int main() {
    RenderWindow window(VideoMode(540, 540), "Sudoku Board");

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

    vector<vector<bool>> fixedCells(N, vector<bool>(N, false));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (board[i][j] != 0) {
                fixedCells[i][j] = true;
            }
        }
    }

    Font font;
    if (!font.loadFromFile("arial.ttf")) {
        cerr << "Could not load font" << endl;
        return -1;
    }

    if (!solveSudoku(board, 0, 0, fixedCells)) {
        cerr << "No solution exists" << endl;
        return -1;
    }

    while (window.isOpen()) {
        Event event;
        while (window.pollEvent(event)) {
            if (event.type == Event::Closed)
                window.close();
        }

        window.clear(Color::White);
        drawGrid(window, board, fixedCells, font);
        window.display();
    }

    return 0;
}
