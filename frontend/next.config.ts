import type { NextConfig } from "next";

const backend = process.env.RESEARCH_API_ORIGIN ?? "http://127.0.0.1:2024";

const nextConfig: NextConfig = {
  output: "standalone",
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  async rewrites() {
    return [{ source: "/api/research/:path*", destination: `${backend}/:path*` }];
  },
};

export default nextConfig;
