import * as React from "react";
import { cn } from "@/lib/utils";

/** ScrollArea without Radix — overflow auto for MVP bundle size. */
function ScrollArea({
  className,
  children,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      className={cn("relative overflow-auto", className)}
      {...props}
    >
      {children}
    </div>
  );
}

export { ScrollArea };
