# Chat History Feature Plan

> Goal: 像 ChatGPT 一样保存历史对话，按日期分组，刷新/关闭页面后可恢复。

---

## Current State

| 维度 | 现状 | 问题 |
|------|------|------|
| Session ID | 前端硬编码 `"demo"` | 所有对话混在一个 session 里 |
| 消息存储 | Redis list per session_id | 后端有数据，但前端从不读取历史 |
| 页面刷新 | JS 状态全丢，聊天区清空 | 无法恢复之前的对话 |
| 会话列表 | 不存在 | 无法切换不同对话 |

---

## Architecture Design

### Storage: 后端 Redis + 前端 localStorage 双层

| 层 | 存储内容 | 生命周期 |
|----|---------|---------|
| **Redis (后端)** | 每个 session 的消息列表（已有） | 服务运行期间 + RDB 持久化 |
| **localStorage (前端)** | 会话索引（id, title, date, preview） | 永久（用户手动清除前） |

为什么不只用一侧：
- 只用 Redis：前端刷新后不知道有哪些 session 存在，需要 API 列举
- 只用 localStorage：消息量大时占用浏览器存储，且不跨设备
- 双层：localStorage 存轻量索引（快速渲染侧边栏），Redis 存完整消息（按需加载）

### 新增 Backend API

| Endpoint | Method | 功能 |
|----------|--------|------|
| `/chat/sessions` | GET | 列出所有 session（id + 消息数 + 最后活跃时间） |
| `/chat/history/{session_id}` | GET | 获取某个 session 的完整消息列表 |
| `/chat/sessions/{session_id}` | DELETE | 删除某个 session 的所有数据 |

### Frontend Components

```
┌─────────────────────────────────────────────────┐
│  header (yang agent)                            │
├──────────┬──────────────────────────────────────┤
│ sidebar  │  toolbar                             │
│          │──────────────────────────────────────│
│ [+ New]  │  upload panel (hidden)               │
│          │──────────────────────────────────────│
│ Today    │                                      │
│  Chat 1  │  chat messages                       │
│  Chat 2  │                                      │
│          │                                      │
│ Yesterday│                                      │
│  Chat 3  │                                      │
│          │                                      │
│ Mar 28   │                                      │
│  Chat 4  │──────────────────────────────────────│
│          │  thinking bar                        │
│          │──────────────────────────────────────│
│          │  footer (input + send)               │
└──────────┴──────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Backend API (3 endpoints)

**Files to modify:**
- `app/api/routes.py` — 新增 3 个端点
- `app/api/schemas.py` — 新增 request/response schemas
- `app/memory/short_term.py` — 新增 `list_sessions()`, `get_all_messages()`, `delete_session()` 方法

**`GET /chat/sessions`** 返回：
```json
{
  "sessions": [
    {
      "session_id": "a1b2c3",
      "message_count": 12,
      "last_active": "2026-03-31T15:30:00",
      "preview": "What is RAG architecture?"
    }
  ]
}
```

实现方式：Redis `SCAN` 匹配 `deepsearch:st:*:messages`，对每个 session 读最后一条消息作 preview。

**`GET /chat/history/{session_id}`** 返回：
```json
{
  "session_id": "a1b2c3",
  "messages": [
    {"role": "user", "content": "...", "timestamp": 1711900000},
    {"role": "assistant", "content": "...", "timestamp": 1711900005}
  ]
}
```

**`DELETE /chat/sessions/{session_id}`** 删除 Redis 中该 session 的 messages + traces + task_state。

### Phase 2: Frontend Sidebar

**File to modify:** `frontend/index.html`

**2.1 Layout 改造**

当前 `.container` 是单列布局。改为：
```css
.container {
  display: grid;
  grid-template-columns: 220px 1fr;  /* sidebar + main */
  grid-template-rows: auto 1fr auto;
}
```

**2.2 Sidebar HTML**
```html
<aside class="sidebar">
  <button class="new-chat-btn">+ New Chat</button>
  <div class="session-list">
    <!-- 按日期分组，JS 动态渲染 -->
  </div>
</aside>
```

**2.3 Session 管理 JS 逻辑**

```
页面加载:
  1. 从 localStorage 读取 sessions 索引
  2. 调 GET /chat/sessions 同步最新状态
  3. 渲染侧边栏（按日期分组：Today / Yesterday / Earlier）
  4. 自动选中最近的 session，调 GET /chat/history/{id} 加载消息

点击 "+ New Chat":
  1. 生成新 session_id (UUID)
  2. 清空聊天区
  3. 更新 localStorage 索引
  4. 高亮新 session

点击已有 session:
  1. 调 GET /chat/history/{id} 获取消息
  2. 渲染到聊天区
  3. 更新当前 SESSION_ID

发送消息后:
  1. 更新 localStorage 中该 session 的 preview 和 last_active
  2. 侧边栏实时更新预览文字

删除 session:
  1. 调 DELETE /chat/sessions/{id}
  2. 从 localStorage 移除
  3. 侧边栏移除该项，自动切换到相邻 session
```

### Phase 3: Session Title Auto-generation

用第一条用户消息的前 30 个字符作为 session title。存在 localStorage 中。

如果后续接入真实 LLM，可以用 LLM 生成更智能的标题（类似 ChatGPT）。

---

## Date Grouping Logic

```javascript
function getDateGroup(timestamp) {
  const now = new Date();
  const date = new Date(timestamp);
  const diffDays = Math.floor((now - date) / 86400000);

  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return "This Week";
  if (diffDays < 30) return "This Month";
  return date.toLocaleDateString();  // "3/15/2026"
}
```

---

## localStorage Schema

```json
{
  "deepsearch_sessions": [
    {
      "session_id": "a1b2c3d4",
      "title": "What is RAG architecture?",
      "created_at": 1711900000000,
      "last_active": 1711903600000,
      "preview": "RAG combines retrieval and generation..."
    }
  ],
  "deepsearch_current_session": "a1b2c3d4"
}
```

---

## Files Changed Summary

| File | Change |
|------|--------|
| `app/memory/short_term.py` | +3 methods: `list_sessions()`, `get_all_messages()`, `delete_session()` |
| `app/api/schemas.py` | +3 schemas: `SessionInfo`, `SessionListResponse`, `ChatHistoryResponse` |
| `app/api/routes.py` | +3 endpoints: GET/DELETE sessions, GET history |
| `frontend/index.html` | Sidebar layout + session switching + localStorage persistence |

---

## Execution Order

```
Phase 1: Backend API          ← 先有数据接口
  ↓
Phase 2: Frontend Sidebar     ← 侧边栏 + 会话切换
  ↓
Phase 3: Auto Title           ← 锦上添花
```
