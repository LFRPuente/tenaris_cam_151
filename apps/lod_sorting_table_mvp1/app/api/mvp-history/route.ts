import { NextResponse } from "next/server"

export const dynamic = "force-dynamic"

const SOURCE_URL = process.env.MVP_SOURCE_URL || "http://127.0.0.1:58597"

function sourceUrl(pathname: string) {
  return new URL(pathname, SOURCE_URL.endsWith("/") ? SOURCE_URL : `${SOURCE_URL}/`)
}

export async function GET() {
  const target = sourceUrl("/api/history")
  try {
    const response = await fetch(target, { cache: "no-store" })
    const text = await response.text()
    if (!response.ok) {
      return NextResponse.json(
        { error: text || `MVP source returned ${response.status}` },
        { status: response.status },
      )
    }
    return new NextResponse(text, {
      headers: {
        "content-type": response.headers.get("content-type") || "application/json",
      },
    })
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Could not load the MVP history.",
        source_url: target.toString(),
      },
      { status: 502 },
    )
  }
}
