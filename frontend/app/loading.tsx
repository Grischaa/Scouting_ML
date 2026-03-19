export default function Loading() {
  return (
    <div className="min-h-screen bg-bg px-4 py-6 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-[1600px] space-y-6">
        <div className="h-28 animate-pulse rounded-[28px] border border-white/8 bg-panel/90" />
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-6">
          {Array.from({ length: 6 }).map((_, index) => (
            <div key={index} className="h-32 animate-pulse rounded-[24px] border border-white/8 bg-panel/90" />
          ))}
        </div>
        <div className="grid gap-5 xl:grid-cols-12">
          <div className="space-y-5 xl:col-span-7">
            <div className="grid gap-5 lg:grid-cols-12">
              <div className="h-[520px] animate-pulse rounded-[24px] border border-white/8 bg-panel/90 lg:col-span-12" />
              <div className="h-80 animate-pulse rounded-[24px] border border-white/8 bg-panel/90 lg:col-span-6" />
              <div className="h-80 animate-pulse rounded-[24px] border border-white/8 bg-panel/90 lg:col-span-6" />
              <div className="h-96 animate-pulse rounded-[24px] border border-white/8 bg-panel/90 lg:col-span-12" />
            </div>
          </div>
          <div className="space-y-5 xl:col-span-5">
            <div className="h-96 animate-pulse rounded-[24px] border border-white/8 bg-panel/90" />
            <div className="h-80 animate-pulse rounded-[24px] border border-white/8 bg-panel/90" />
            <div className="h-80 animate-pulse rounded-[24px] border border-white/8 bg-panel/90" />
          </div>
        </div>
      </div>
    </div>
  );
}
