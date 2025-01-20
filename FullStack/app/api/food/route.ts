import { NextResponse } from 'next/server';

const EDAMAM_BASE_URL = 'https://api.edamam.com/api/food-database/v2/parser';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const query = searchParams.get('query');

    if (!query) {
      return NextResponse.json(
        { error: 'Query parameter is required' },
        { status: 400 }
      );
    }

    const appId = process.env.NEXT_PUBLIC_EDAMAM_APP_ID;
    const appKey = process.env.NEXT_PUBLIC_EDAMAM_APP_KEY;

    if (!appId || !appKey) {
      return NextResponse.json(
        { error: 'API configuration is missing' },
        { status: 500 }
      );
    }

    const edamamUrl = `${EDAMAM_BASE_URL}?app_id=${appId}&app_key=${appKey}&ingr=${encodeURIComponent(query)}`;

    const response = await fetch(edamamUrl, {
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Edamam API returned ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('Food search error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch food data' },
      { status: 500 }
    );
  }
}