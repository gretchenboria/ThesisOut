const ORIGIN = "https://thesisout-421717890494.us-central1.run.app";

const BAD_BOT_PATTERNS = [
  "python-requests",
  "scrapy",
  "headlesschrome",
  "phantomjs",
  "selenium",
  "puppeteer",
  "playwright",
  "gptbot",
  "ccbot",
  "bytespider",
  "claudebot",
  "ahrefsbot",
  "semrushbot",
  "mj12bot",
  "dotbot",
  "petalbot",
];

export default {
  async fetch(request) {
    const ua = (request.headers.get("User-Agent") || "").toLowerCase();
    if (!ua || BAD_BOT_PATTERNS.some((p) => ua.includes(p))) {
      return new Response("Access Denied: Automated bot traffic is blocked.", {
        status: 403,
        headers: { "Content-Type": "text/plain" },
      });
    }

    const url = new URL(request.url);
    const targetUrl = new URL(url.pathname + url.search, ORIGIN);

    const headers = new Headers(request.headers);
    headers.set("Host", "thesisout-421717890494.us-central1.run.app");
    headers.set("X-Forwarded-Host", url.host);
    headers.set("X-Forwarded-Proto", url.protocol.replace(":", ""));

    const init = {
      method: request.method,
      headers,
      redirect: "manual",
    };

    if (request.method !== "GET" && request.method !== "HEAD") {
      init.body = request.body;
    }

    const response = await fetch(targetUrl.toString(), init);
    const respHeaders = new Headers(response.headers);
    respHeaders.set("X-Robots-Tag", "noindex, nofollow, noarchive");

    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: respHeaders,
    });
  },
};
