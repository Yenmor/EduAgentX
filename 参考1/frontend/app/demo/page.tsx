import { redirect } from "next/navigation";

export default function DemoPage() {
  redirect("/classroom?sessionId=demo");
}
