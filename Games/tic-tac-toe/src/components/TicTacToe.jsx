import React, { useState, useEffect } from 'react';
import './TicTacToe.css';

const TicTacToe = () => {
  const [board, setBoard] = useState(Array(9).fill(null));
  const [isXNext, setIsXNext] = useState(true);
  const [gameMode, setGameMode] = useState(null);
  const [gameStatus, setGameStatus] = useState('');
  const [isGameActive, setIsGameActive] = useState(true);

  const winningCombos = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // columns
    [0, 4, 8], [2, 4, 6] // diagonals
  ];

  const checkWinner = (squares) => {
    for (const combo of winningCombos) {
      const [a, b, c] = combo;
      if (squares[a] && squares[a] === squares[b] && squares[a] === squares[c]) {
        return { winner: squares[a], line: combo };
      }
    }
    return null;
  };

  const isBoardFull = (squares) => squares.every(square => square !== null);

  const computerMove = (currentBoard) => {
    // Try to win
    const winMove = findBestMove(currentBoard, 'O');
    if (winMove !== -1) return winMove;

    // Block player
    const blockMove = findBestMove(currentBoard, 'X');
    if (blockMove !== -1) return blockMove;

    // Take center if available
    if (!currentBoard[4]) return 4;

    // Take corners
    const corners = [0, 2, 6, 8];
    const availableCorners = corners.filter(i => !currentBoard[i]);
    if (availableCorners.length > 0) {
      return availableCorners[Math.floor(Math.random() * availableCorners.length)];
    }

    // Take any available space
    const availableSpaces = currentBoard
      .map((square, index) => !square ? index : null)
      .filter(space => space !== null);

    return availableSpaces[Math.floor(Math.random() * availableSpaces.length)];
  };

  const findBestMove = (currentBoard, player) => {
    for (const combo of winningCombos) {
      const [a, b, c] = combo;
      const line = [currentBoard[a], currentBoard[b], currentBoard[c]];
      const playerCount = line.filter(square => square === player).length;
      const emptyCount = line.filter(square => !square).length;

      if (playerCount === 2 && emptyCount === 1) {
        const emptyIndex = combo[line.findIndex(square => !square)];
        return emptyIndex;
      }
    }
    return -1;
  };

  useEffect(() => {
    if (gameMode === 'computer' && !isXNext && isGameActive) {
      const timer = setTimeout(() => {
        makeComputerMove();
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [isXNext, gameMode, isGameActive]);

  const makeComputerMove = () => {
    const newBoard = [...board];
    const move = computerMove(newBoard);
    
    if (move !== -1) {
      newBoard[move] = 'O';
      setBoard(newBoard);
      
      const result = checkWinner(newBoard);
      if (result) {
        setGameStatus(`Computer Wins!`);
        setIsGameActive(false);
      } else if (isBoardFull(newBoard)) {
        setGameStatus("It's a Draw!");
        setIsGameActive(false);
      } else {
        setIsXNext(true);
      }
    }
  };

  const handleClick = (index) => {
    if (!isGameActive || board[index] || (gameMode === 'computer' && !isXNext)) {
      return;
    }

    const newBoard = [...board];
    newBoard[index] = isXNext ? 'X' : 'O';
    setBoard(newBoard);

    const result = checkWinner(newBoard);
    if (result) {
      setGameStatus(`${result.winner} Wins!`);
      setIsGameActive(false);
    } else if (isBoardFull(newBoard)) {
      setGameStatus("It's a Draw!");
      setIsGameActive(false);
    } else {
      setIsXNext(!isXNext);
    }
  };

  const resetGame = () => {
    setBoard(Array(9).fill(null));
    setIsXNext(true);
    setGameStatus('');
    setIsGameActive(true);
  };

  return (
    <div className="game-container">
      <h1 className="game-title">Tic Tac Toe</h1>
      
      {!gameMode ? (
        <div className="mode-selection">
          <h2>Choose Your Mode</h2>
          <div className="mode-buttons">
            <button 
              className="mode-button computer"
              onClick={() => setGameMode('computer')}
            >
              Play vs Computer
            </button>
            <button 
              className="mode-button friend"
              onClick={() => setGameMode('friend')}
            >
              Play vs Friend
            </button>
          </div>
        </div>
      ) : (
        <div className="game-content">
          <div className="status-bar">
            <p className="game-mode">Mode: {gameMode === 'computer' ? 'vs Computer' : 'vs Friend'}</p>
            <p className="game-status">
              {gameStatus || `Next Player: ${isXNext ? 'X' : 'O'}`}
            </p>
          </div>

          <div className="board">
            {board.map((value, index) => (
              <button
                key={index}
                className={`cell ${value ? 'filled' : ''} ${value === 'X' ? 'x' : 'o'}`}
                onClick={() => handleClick(index)}
              >
                {value}
              </button>
            ))}
          </div>

          <div className="control-buttons">
            <button className="control-button reset" onClick={resetGame}>
              Reset Game
            </button>
            <button 
              className="control-button change-mode"
              onClick={() => {
                setGameMode(null);
                resetGame();
              }}
            >
              Change Mode
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default TicTacToe;