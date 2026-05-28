import { NextRequest, NextResponse } from "next/server"

export const dynamic = "force-dynamic"

const SOURCE_URL = process.env.MVP_SOURCE_URL || "http://127.0.0.1:58597"

function sourceUrl(pathname: string) {
  return new URL(pathname, SOURCE_URL.endsWith("/") ? SOURCE_URL : `${SOURCE_URL}/`)
}

export async function GET(request: NextRequest) {
  const assetPath = request.nextUrl.searchParams.get("path")
  if (!assetPath || !assetPath.startsWith("/asset/")) {
    return NextResponse.json({ error: "Missing MVP asset path." }, { status: 400 })
  }

  const target = sourceUrl(assetPath)
  try {
    const response = await fetch(target, { cache: "no-store" })
    if (!response.ok) {
      return NextResponse.json(
        { error: `MVP asset returned ${response.status}` },
        { status: response.status },
      )
    }
    return new NextResponse(await response.arrayBuffer(), {
      headers: {
        "content-type": response.headers.get("content-type") || "application/octet-stream",
        "cache-control": "no-store",
      },
    })
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Could not load the MVP asset.",
        source_url: target.toString(),
      },
      { status: 502 },
    )
  }
}
