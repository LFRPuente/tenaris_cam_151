import { NextRequest, NextResponse } from "next/server"

export const dynamic = "force-dynamic"

const SOURCE_URL = process.env.MVP_SOURCE_URL || "http://127.0.0.1:58597"

function sourceUrl(pathname: string) {
  return new URL(pathname, SOURCE_URL.endsWith("/") ? SOURCE_URL : `${SOURCE_URL}/`)
}

function extractInitialState(html: string) {
  const match = html.match(/const INITIAL\s*=\s*(\{[\s\S]*?\});\s*const SVG_NS/)
  if (!match) {
    throw new Error("Could not find the MVP initial state in the source page.")
  }
  return JSON.parse(match[1])
}

export async function GET(request: NextRequest) {
  const runId = request.nextUrl.searchParams.get("run_id")
  const artifact = request.nextUrl.searchParams.get("artifact")
  const target = sourceUrl("/")
  if (artifact) {
    target.searchParams.set("artifact", artifact)
  } else if (runId) {
    target.searchParams.set("run_id", runId)
  }

  try {
    const response = await fetch(target, { cache: "no-store" })
    const text = await response.text()
    if (!response.ok) {
      return NextResponse.json(
        { error: text || `MVP source returned ${response.status}` },
        { status: response.status },
      )
    }
    return NextResponse.json(extractInitialState(text))
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Could not load the MVP source state.",
        source_url: target.toString(),
      },
      { status: 502 },
    )
  }
}
