// app/stocks/page.tsx
"use client";

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import StockCard from '../../components/StockCard';
import { Circles } from 'react-loader-spinner';

const StocksPage = () => {
  const [stocks, setStocks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchStocks = async () => {
      try {
        const response = await axios.get('https://www.alphavantage.co/query', {
          params: {
            function: 'TIME_SERIES_INTRADAY',
            symbol: 'AAPL', // Change this symbol as needed
            interval: '1min',
            apikey: 'ZU45KMJDQF2Y30OD'
          }
        });

        if (response.data['Time Series (1min)']) {
          const timeSeries = response.data['Time Series (1min)'];
          const stockData = Object.keys(timeSeries).map((timestamp) => ({
            timestamp,
            symbol: response.data['Meta Data']['2. Symbol'],
            open: timeSeries[timestamp]['1. open'],
            high: timeSeries[timestamp]['2. high'],
            low: timeSeries[timestamp]['3. low'],
            close: timeSeries[timestamp]['4. close'],
            volume: timeSeries[timestamp]['5. volume'],
          }));
          setStocks(stockData);
        } else {
          throw new Error('Unexpected API response format');
        }

        setLoading(false);
      } catch (err) {
        setError(err);
        setLoading(false);
      }
    };
    fetchStocks();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen bg-gradient-to-b from-black to-purple-900">
        <Circles height="80" width="80" color="#4fa94d" ariaLabel="loading" />
      </div>
    );
  }

  if (error) {
    return <p className="text-center mt-4">Error: {error.message}</p>;
  }

  return (
    <div className="min-h-screen bg-black text-white p-8">
      <h1 className="text-3xl font-bold text-center mb-6">Stock Data</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {stocks.slice(0, 10).map((stock, index) => (
          <StockCard key={index} stock={stock} />
        ))}
      </div>
    </div>
  );
};

export default StocksPage;
