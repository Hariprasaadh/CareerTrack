"use client"

import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Plus, Pizza, X, Scale, Activity, Utensils } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import ParticlesComponent from "@/components/Particles";

interface EdamamNutrient {
  label: string;
  quantity: number;
  unit: string;
}

interface EdamamNutrients {
  ENERC_KCAL: EdamamNutrient;
  PROCNT: EdamamNutrient;
  FAT: EdamamNutrient;
  CHOCDF: EdamamNutrient;
  FIBTG: EdamamNutrient;
  SUGAR: EdamamNutrient;
}

interface EdamamResponse {
  calories: number;
  totalWeight: number;
  dietLabels: string[];
  healthLabels: string[];
  totalNutrients: EdamamNutrients;
}

interface NutrientInfo {
  calories: number;
  carbs: number;
  protein: number;
  fat: number;
  fiber: number;
  sugar: number;
}

interface FoodItem extends NutrientInfo {
  food: string;
  time: string;
  dietLabels?: string[];
  healthLabels?: string[];
}

interface DashboardState {
  meals: FoodItem[];
  searchQuery: string;
  searchResults: FoodItem[];
  showAddForm: boolean;
  loading: boolean;
  error: string | null;
  height: string;
  weight: string;
  bmi: number | null;
  dailyTotals: NutrientInfo;
}

const NutritionDashboard = () => {
  const [state, setState] = useState<DashboardState>({
    meals: [],
    searchQuery: "",
    searchResults: [],
    showAddForm: false,
    loading: false,
    error: null,
    height: "",
    weight: "",
    bmi: null,
    dailyTotals: {
      calories: 0,
      carbs: 0,
      protein: 0,
      fat: 0,
      fiber: 0,
      sugar: 0,
    },
  });

  const formatNutrient = (value: number): string => {
    return Math.round(value).toString();
  };

  const calculateBMI = (height: number, weight: number): number => {
    const heightInMeters = height / 100;
    return weight / (heightInMeters * heightInMeters);
  };

  const getBMICategory = (bmiValue: number) => {
    if (bmiValue < 18.5)
      return { category: "Underweight", color: "text-blue-600" };
    if (bmiValue < 25)
      return { category: "Normal weight", color: "text-green-600" };
    if (bmiValue < 30)
      return { category: "Overweight", color: "text-yellow-600" };
    return { category: "Obese", color: "text-red-600" };
  };

  const analyzeFoodItem = async (query: string): Promise<void> => {
    if (!query) return;

    setState((prev) => ({ ...prev, loading: true, error: null }));

    try {
      const APP_ID = process.env.NEXT_PUBLIC_EDAMAM_APP_ID;
      const APP_KEY = process.env.NEXT_PUBLIC_EDAMAM_APP_KEY;

      if (!APP_ID || !APP_KEY) {
        throw new Error("API credentials are not properly configured");
      }

      const response = await fetch(
        `https://api.edamam.com/api/nutrition-data?app_id=${APP_ID}&app_key=${APP_KEY}&ingr=${encodeURIComponent(query)}`,
        {
          headers: {
            'Accept': 'application/json',
          },
        }
      );

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data: EdamamResponse = await response.json();

      if (!data.calories) {
        setState((prev) => ({
          ...prev,
          searchResults: [],
          loading: false,
          error: "No nutritional information found for your search.",
        }));
        return;
      }

      const foodItem: FoodItem = {
        food: query,
        calories: data.calories,
        carbs: data.totalNutrients.CHOCDF?.quantity || 0,
        protein: data.totalNutrients.PROCNT?.quantity || 0,
        fat: data.totalNutrients.FAT?.quantity || 0,
        fiber: data.totalNutrients.FIBTG?.quantity || 0,
        sugar: data.totalNutrients.SUGAR?.quantity || 0,
        time: new Date().toLocaleTimeString(),
        dietLabels: data.dietLabels,
        healthLabels: data.healthLabels,
      };

      setState((prev) => ({
        ...prev,
        searchResults: [foodItem],
        loading: false,
        error: null,
      }));
    } catch (err) {
      console.error('Analyze Food Error:', err);
      setState((prev) => ({
        ...prev,
        error: err instanceof Error ? err.message : "Failed to analyze food. Please try again.",
        loading: false,
        searchResults: [],
      }));
    }
  };

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setState((prev) => ({ ...prev, searchQuery: event.target.value }));
  };

  const handleFoodSelect = (foodItem: FoodItem) => {
    setState((prev) => {
      const updatedMeals = [...prev.meals, foodItem];
      
      const newTotals = updatedMeals.reduce(
        (acc, meal) => ({
          calories: acc.calories + meal.calories,
          carbs: acc.carbs + meal.carbs,
          protein: acc.protein + meal.protein,
          fat: acc.fat + meal.fat,
          fiber: acc.fiber + meal.fiber,
          sugar: acc.sugar + meal.sugar,
        }),
        {
          calories: 0,
          carbs: 0,
          protein: 0,
          fat: 0,
          fiber: 0,
          sugar: 0,
        }
      );

      return {
        ...prev,
        meals: updatedMeals,
        showAddForm: false,
        searchResults: [],
        searchQuery: "",
        dailyTotals: newTotals,
      };
    });
  };

  const handleBMICalculation = () => {
    if (state.height && state.weight) {
      const heightNum = parseFloat(state.height);
      const weightNum = parseFloat(state.weight);
      
      if (isNaN(heightNum) || isNaN(weightNum) || heightNum <= 0 || weightNum <= 0) {
        setState(prev => ({ ...prev, error: "Please enter valid height and weight values" }));
        return;
      }
      
      const bmiValue = calculateBMI(heightNum, weightNum);
      setState((prev) => ({ ...prev, bmi: parseFloat(bmiValue.toFixed(1)), error: null }));
    } else {
      setState(prev => ({ ...prev, error: "Please enter both height and weight" }));
    }
  };

  useEffect(() => {
    let isActive = true;
    const timeoutId = setTimeout(() => {
      if (state.searchQuery && isActive) {
        analyzeFoodItem(state.searchQuery);
      }
    }, 500);

    return () => {
      isActive = false;
      clearTimeout(timeoutId);
    };
  }, [state.searchQuery]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <ParticlesComponent />
      <div className="max-w-7xl mx-auto p-4 sm:p-6 lg:p-8 space-y-8">
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">
            Nutrition Dashboard
          </h1>
          <p className="text-gray-600 dark:text-gray-400">Track your daily nutrition goals</p>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          <Card className="bg-white/50 backdrop-blur-lg shadow-xl dark:bg-gray-800/50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Scale className="w-6 h-6 text-blue-500" />
                BMI Calculator
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Height (cm)</label>
                  <input
                    type="number"
                    value={state.height}
                    onChange={(e) => setState((prev) => ({ ...prev, height: e.target.value }))}
                    className="w-full p-3 border rounded-lg bg-white/80 dark:bg-gray-700/50"
                    placeholder="Height in cm"
                    min="1"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Weight (kg)</label>
                  <input
                    type="number"
                    value={state.weight}
                    onChange={(e) => setState((prev) => ({ ...prev, weight: e.target.value }))}
                    className="w-full p-3 border rounded-lg bg-white/80 dark:bg-gray-700/50"
                    placeholder="Weight in kg"
                    min="1"
                  />
                </div>
              </div>
              <button
                onClick={handleBMICalculation}
                className="w-full mt-4 p-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg 
                hover:from-blue-600 hover:to-blue-700 transition-all duration-200 shadow-lg"
              >
                Calculate BMI
              </button>
              {state.error && (
                <Alert className="mt-4">
                  <AlertDescription>{state.error}</AlertDescription>
                </Alert>
              )}
              {state.bmi && (
                <div className="mt-4 p-4 bg-white/80 dark:bg-gray-700/50 rounded-lg">
                  <p className="text-lg">
                    Your BMI: <span className="font-bold">{state.bmi}</span>
                    {" - "}
                    <span className={getBMICategory(state.bmi).color}>
                      {getBMICategory(state.bmi).category}
                    </span>
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-white/50 backdrop-blur-lg shadow-xl dark:bg-gray-800/50">
            <CardContent className="pt-6">
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={state.meals}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'rgba(255, 255, 255, 0.9)',
                        borderRadius: '8px',
                        border: 'none',
                        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
                      }}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="calories" stroke="#10B981" strokeWidth={2} />
                    <Line type="monotone" dataKey="carbs" stroke="#3B82F6" strokeWidth={2} />
                    <Line type="monotone" dataKey="protein" stroke="#EF4444" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="bg-gradient-to-br from-green-500/10 to-green-600/10 shadow-lg">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <Activity className="w-8 h-8 text-green-500" />
                <div>
                  <p className="text-sm font-medium text-green-600">Total Calories</p>
                  <p className="text-2xl font-bold text-green-700">
                    {formatNutrient(state.dailyTotals.calories)}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-gradient-to-br from-blue-500/10 to-blue-600/10 shadow-lg">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <Activity className="w-8 h-8 text-blue-500" />
                <div>
                  <p className="text-sm font-medium text-blue-600">Carbs (g)</p>
                  <p className="text-2xl font-bold text-blue-700">
                    {formatNutrient(state.dailyTotals.carbs)}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-gradient-to-br from-red-500/10 to-red-600/10 shadow-lg">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <Activity className="w-8 h-8 text-red-500" />
                <div>
                  <p className="text-sm font-medium text-red-600">Protein (g)</p>
                  <p className="text-2xl font-bold text-red-700">
                    {formatNutrient(state.dailyTotals.protein)}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-gradient-to-br from-purple-500/10 to-purple-600/10 shadow-lg">
            <CardContent className="p-6">
              <div className="flex items-center gap-4">
                <Activity className="w-8 h-8 text-purple-500" />
                <div>
                  <p className="text-sm font-medium text-purple-600">Fat (g)</p>
                  <p className="text-2xl font-bold text-purple-700">
                    {formatNutrient(state.dailyTotals.fat)}
                    </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <Card className="bg-white/50 backdrop-blur-lg shadow-xl dark:bg-gray-800/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Utensils className="w-6 h-6 text-green-500" />
              Today's Meals
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {state.meals.map((meal, index) => (
                <div
                  key={index}
                  className="p-4 bg-white/80 dark:bg-gray-700/50 rounded-lg shadow-sm hover:shadow-md 
                  transition-shadow duration-200"
                >
                  <div className="flex items-center justify-between flex-wrap gap-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-orange-100 rounded-full">
                        <Pizza className="w-5 h-5 text-orange-500" />
                      </div>
                      <span className="font-medium">{meal.food}</span>
                    </div>
                    <div className="text-gray-600 dark:text-gray-300 text-sm">
                      {formatNutrient(meal.calories)} cal | {formatNutrient(meal.carbs)}g carbs |{" "}
                      {formatNutrient(meal.protein)}g protein | {formatNutrient(meal.fat)}g fat
                    </div>
                  </div>
                  {meal.dietLabels && meal.dietLabels.length > 0 && (
                    <div className="flex gap-2 mt-3">
                      {meal.dietLabels.map((label, i) => (
                        <span
                          key={i}
                          className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded-full"
                        >
                          {label}
                        </span>
                      ))}
                    </div>
                  )}
                  {meal.healthLabels && meal.healthLabels.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-2">
                      {meal.healthLabels.slice(0, 3).map((label, i) => (
                        <span
                          key={i}
                          className="text-xs px-2 py-1 bg-green-100 text-green-800 rounded-full"
                        >
                          {label}
                        </span>
                      ))}
                      {meal.healthLabels.length > 3 && (
                        <span className="text-xs text-gray-500">
                          +{meal.healthLabels.length - 3} more
                        </span>
                      )}
                    </div>
                  )}
                </div>
              ))}
              {state.meals.length === 0 && (
                <div className="text-center text-gray-500 py-8">
                  <Pizza className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                  <p>No meals added yet. Click the + button to add your first meal.</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Add Food Button */}
      <button
        onClick={() => setState((prev) => ({ ...prev, showAddForm: true }))}
        className="fixed bottom-8 right-8 bg-gradient-to-r from-green-500 to-green-600 text-white p-4 
        rounded-full shadow-lg hover:shadow-xl hover:from-green-600 hover:to-green-700 
        transition-all duration-200 transform hover:scale-105"
      >
        <Plus className="w-6 h-6" />
      </button>

      {/* Add Food Modal */}
      {state.showAddForm && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl w-full max-w-md">
            <div className="p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold">Add Food</h2>
                <button
                  onClick={() => setState((prev) => ({ ...prev, showAddForm: false }))}
                  className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>
              <input
                type="text"
                value={state.searchQuery}
                onChange={handleSearchChange}
                placeholder="Enter food (e.g., '1 large apple' or '100g chicken')"
                className="w-full p-4 border rounded-lg mb-4 bg-gray-50 dark:bg-gray-700"
              />

              {state.loading && (
                <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-gray-500 border-t-transparent"></div>
                  Analyzing...
                </div>
              )}
              
              {state.error && (
                <Alert variant="destructive">
                  <AlertDescription>{state.error}</AlertDescription>
                </Alert>
              )}

              <div className="max-h-96 overflow-y-auto">
                {state.searchResults.map((item, index) => (
                  <div
                    key={index}
                    onClick={() => handleFoodSelect(item)}
                    className="p-4 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer rounded-lg 
                    transition-colors duration-200 mb-2"
                  >
                    <div className="font-medium">{item.food}</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      {formatNutrient(item.calories)} cal | {formatNutrient(item.protein)}g protein |{" "}
                      {formatNutrient(item.carbs)}g carbs
                    </div>
                    {item.dietLabels && item.dietLabels.length > 0 && (
                      <div className="flex gap-2 mt-2">
                        {item.dietLabels.map((label, i) => (
                          <span
                            key={i}
                            className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded-full"
                          >
                            {label}
                          </span>
                        ))}
                      </div>
                    )}
                    {item.healthLabels && item.healthLabels.length > 0 && (
                      <div className="flex flex-wrap gap-2 mt-2">
                        {item.healthLabels.slice(0, 3).map((label, i) => (
                          <span
                            key={i}
                            className="text-xs px-2 py-1 bg-green-100 text-green-800 rounded-full"
                          >
                            {label}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Loading Overlay */}
      {state.loading && (
        <div className="fixed inset-0 bg-black/20 backdrop-blur-sm flex items-center justify-center">
          <div className="bg-white p-6 rounded-lg shadow-xl">
            <div className="flex items-center gap-3">
              <div className="animate-spin rounded-full h-6 w-6 border-2 border-green-500 border-t-transparent"></div>
              <p className="text-gray-700">Analyzing nutrition data...</p>
            </div>
          </div>
        </div>
      )}

      {/* Error Toast */}
      {state.error && (
        <div className="fixed bottom-4 left-4 bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded shadow-lg">
          <div className="flex">
            <div className="flex-shrink-0">
              <X className="h-5 w-5 text-red-500" />
            </div>
            <div className="ml-3">
              <p className="text-sm">{state.error}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NutritionDashboard;