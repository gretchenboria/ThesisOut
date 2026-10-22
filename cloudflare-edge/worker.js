const ORIGIN = "https://thesisout-421717890494.us-central1.run.app";

export default {
  async fetch(request) {
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

    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: respHeaders,
    });
  },
};
