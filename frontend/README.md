# ScoutML Frontend

Premium football scouting and player analytics frontend built with:

- Next.js
- React
- TypeScript
- Tailwind CSS
- Framer Motion
- Recharts
- lucide-react

## Run locally

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## Structure

- `app/` App Router pages and layouts
- `components/` Reusable UI and domain components
- `lib/mock-data.ts` Mock scouting dataset used across all pages
- `lib/types.ts` Shared TypeScript types

## Notes

- This frontend is mock-data-only by design.
- No backend integration is required to demo the product.
- The visual direction is dark-first and optimized for football recruitment workflows rather than generic BI dashboards.
