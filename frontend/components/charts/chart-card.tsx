import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { SectionHeader } from "@/components/ui/section-header";

export function ChartCard({
  title,
  description,
  action,
  children,
}: {
  title: string;
  description?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <Card className="h-full">
      <CardHeader>
        <SectionHeader title={title} description={description} action={action} />
      </CardHeader>
      <CardContent className="pt-4">{children}</CardContent>
    </Card>
  );
}
