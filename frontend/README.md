# Open Deep Research Console

## 本地开发

先在仓库根目录启动 FastAPI：

```powershell
uvicorn open_deep_research.server:app --reload --host 127.0.0.1 --port 2024
```

再启动前端：

```powershell
cd frontend
Copy-Item .env.example .env.local
pnpm install
pnpm dev
```

浏览器访问 `http://localhost:3000`。本地旁路需要前端 `NEXT_PUBLIC_LOCAL_DEV_AUTH_BYPASS=true`，同时后端环境设置 `LOCAL_DEV_AUTH_BYPASS=true`。

## 验证

```powershell
pnpm test
pnpm build
pnpm test:e2e
```

生产构建使用 `/api/research` 同源前缀。仓库根目录的 `compose.demo.yml` 提供 Next.js、FastAPI 和关闭 SSE 缓冲的 Nginx 演示拓扑。
