import { NextResponse } from "next/server"

export const dynamic = "force-dynamic"

const SOURCE_URL = process.env.MVP_SOURCE_URL || "http://127.0.0.1:58597"

function sourceUrl(pathname: string) {
  return new URL(pathname, SOURCE_URL.endsWith("/") ? SOURCE_URL : `${SOURCE_URL}/`)
}

export async function POST() {
  const target = sourceUrl("/api/capture")
  try {
    const response = await fetch(target, {
      cache: "no-store",
      method: "POST",
    })
    const text = await response.text()
    const contentType = response.headers.get("content-type") || "application/json"
    if (!response.ok) {
      let error = text || `MVP source returned ${response.status}`
      try {
        error = JSON.parse(text).error || error
      } catch {
        // Keep the plain-text response.
      }
      return NextResponse.json({ error, source_url: target.toString() }, { status: response.status })
    }
    return new NextResponse(text, {
      headers: {
        "content-type": contentType,
      },
    })
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Could not start the capture job.",
        source_url: target.toString(),
      },
      { status: 502 },
    )
  }
}
