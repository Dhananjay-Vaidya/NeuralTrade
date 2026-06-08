import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/providers";
import { Sidebar } from "@/components/layout/Sidebar";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "NeuralTrade — AI Backtesting Research",
  description:
    "Backtested RL trading research tool. Not financial advice. Past performance does not guarantee future results.",
  keywords: ["trading", "backtesting", "reinforcement learning", "research"],
  openGraph: {
    title: "NeuralTrade — AI Backtesting Research",
    description: "Backtested RL trading research. Not financial advice.",
    type: "website",
  },
  robots: "noindex, nofollow",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} h-full`}
    >
      <body className="flex h-full antialiased">
        <Providers>
          {/* App shell */}
          <div className="flex h-full w-full overflow-hidden">
            <Sidebar />
            <div className="flex flex-1 flex-col overflow-hidden">
              {children}
            </div>
          </div>
        </Providers>
      </body>
    </html>
  );
}
