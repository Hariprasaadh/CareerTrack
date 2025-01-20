import React, { useState, useEffect } from 'react';
import './Game2048.css';

const Game2048 = () => {
  const [board, setBoard] = useState(getInitialBoard());
  const [score, setScore] = useState(0);
  const [gameOver, setGameOver] = useState(false);
  const [won, setWon] = useState(false);

  // Initialize empty board and add 2 random tiles
  function getInitialBoard() {
    const board = Array(4).fill().map(() => Array(4).fill(0));
    return addRandomTile(addRandomTile(board));
  }

  // Add a new tile (2 or 4) to random empty cell
  function addRandomTile(board) {
    const emptyCells = [];
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        if (board[i][j] === 0) {
          emptyCells.push([i, j]);
        }
      }
    }
    
    if (emptyCells.length > 0) {
      const [row, col] = emptyCells[Math.floor(Math.random() * emptyCells.length)];
      const newBoard = board.map(row => [...row]);
      newBoard[row][col] = Math.random() < 0.9 ? 2 : 4;
      return newBoard;
    }
    return board;
  }

  // Move tiles in specified direction
  function moveTiles(direction) {
    if (gameOver || won) return;

    let newBoard = board.map(row => [...row]);
    let moved = false;
    let addedScore = 0;

    // Helper function to merge tiles
    const mergeTiles = (line) => {
      // Remove zeros
      line = line.filter(cell => cell !== 0);
      
      // Merge adjacent same numbers
      for (let i = 0; i < line.length - 1; i++) {
        if (line[i] === line[i + 1]) {
          line[i] *= 2;
          addedScore += line[i];
          if (line[i] === 2048) setWon(true);
          line.splice(i + 1, 1);
        }
      }
      
      // Add zeros back
      while (line.length < 4) {
        line.push(0);
      }
      
      return line;
    };

    // Process each row/column based on direction
    if (direction === 'left' || direction === 'right') {
      newBoard = newBoard.map(row => {
        const oldRow = [...row];
        const newRow = direction === 'left' ? 
          mergeTiles([...row]) : 
          mergeTiles([...row].reverse()).reverse();
        
        if (JSON.stringify(oldRow) !== JSON.stringify(newRow)) {
          moved = true;
        }
        return newRow;
      });
    } else {
      // For up/down, transpose the board, merge, then transpose back
      let rotatedBoard = Array(4).fill().map((_, i) => 
        Array(4).fill().map((_, j) => newBoard[j][i])
      );
      
      rotatedBoard = rotatedBoard.map(row => {
        const oldRow = [...row];
        const newRow = direction === 'up' ? 
          mergeTiles([...row]) : 
          mergeTiles([...row].reverse()).reverse();
        
        if (JSON.stringify(oldRow) !== JSON.stringify(newRow)) {
          moved = true;
        }
        return newRow;
      });
      
      newBoard = Array(4).fill().map((_, i) => 
        Array(4).fill().map((_, j) => rotatedBoard[j][i])
      );
    }

    if (moved) {
      newBoard = addRandomTile(newBoard);
      setBoard(newBoard);
      setScore(score + addedScore);
      
      // Check for game over
      if (isBoardFull(newBoard) && !hasValidMoves(newBoard)) {
        setGameOver(true);
      }
    }
  }

  // Check if board is full
  function isBoardFull(board) {
    return !board.some(row => row.some(cell => cell === 0));
  }

  // Check if any valid moves remain
  function hasValidMoves(board) {
    // Check for adjacent same numbers
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        if (j < 3 && board[i][j] === board[i][j + 1]) return true;
        if (i < 3 && board[i][j] === board[i + 1][j]) return true;
      }
    }
    return false;
  }

  // Handle keyboard events
  useEffect(() => {
    const handleKeyDown = (e) => {
      switch(e.key) {
        case 'ArrowLeft':
          moveTiles('left');
          break;
        case 'ArrowRight':
          moveTiles('right');
          break;
        case 'ArrowUp':
          moveTiles('up');
          break;
        case 'ArrowDown':
          moveTiles('down');
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [board, gameOver, won]);

  // Reset game
  const resetGame = () => {
    setBoard(getInitialBoard());
    setScore(0);
    setGameOver(false);
    setWon(false);
  };

  return (
    <div className="game-container">
      <div className="game-header">
        <div className="game-title">2048</div>
        <div className="score-box">Score: {score}</div>
        <button className="new-game-btn" onClick={resetGame}>
          New Game
        </button>
      </div>

      <div className="board-container">
        <div className="game-grid">
          {board.map((row, i) => 
            row.map((cell, j) => (
              <div 
                key={`${i}-${j}`}
                className={`game-tile ${cell === 0 ? 'tile-empty' : `tile-${cell}`}`}
              >
                {cell !== 0 && cell}
              </div>
            ))
          )}
        </div>
      </div>

      {(gameOver || won) && (
        <div className="game-message">
          {won ? "Congratulations! You won!" : "Game Over!"}
        </div>
      )}

      <div className="game-instructions">
        Use arrow keys to move tiles
      </div>
    </div>
  );
};

export default Game2048;