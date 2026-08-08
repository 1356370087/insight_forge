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

浏览器访问 `http://localhost:3000`。常规开发应先配置 PostgreSQL、执行 `alembic upgrade head`，再用 `python -m security.cli bootstrap-admin` 创建首位管理员。只有明确的无认证本地演示才同时设置前端 `NEXT_PUBLIC_LOCAL_DEV_AUTH_BYPASS=true` 和后端 `LOCAL_DEV_AUTH_BYPASS=true`。

浏览器只连接 Next.js BFF。Access Token 与 Refresh Token 使用 HttpOnly Cookie 保存，研究请求由 `/api/research` 代理注入身份；不要在 `NEXT_PUBLIC_*` 变量或 localStorage 中存放任何 JWT。

## 验证

```powershell
pnpm test
pnpm build
pnpm test:e2e
```

生产构建使用 `/api/research` 同源前缀。仓库根目录的 `compose.demo.yml` 提供 Next.js、FastAPI 和关闭 SSE 缓冲的 Nginx 演示拓扑。
