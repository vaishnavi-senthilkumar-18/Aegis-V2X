import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Overview } from "./pages/Overview";
import { DigitalTwin } from "./pages/DigitalTwin";
import { V2xNetwork } from "./pages/V2xNetwork";
import { TwinTrustAp } from "./pages/TwinTrustAp";
import { Simulation } from "./pages/Simulation";
import { Experiments } from "./pages/Experiments";
import { Analytics } from "./pages/Analytics";
import { SystemHealth } from "./pages/SystemHealth";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 2000,
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename="/dashboard">
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Overview />} />
            <Route path="digital-twin" element={<DigitalTwin />} />
            <Route path="v2x-network" element={<V2xNetwork />} />
            <Route path="twintrust-ap" element={<TwinTrustAp />} />
            <Route path="simulation" element={<Simulation />} />
            <Route path="experiments" element={<Experiments />} />
            <Route path="analytics" element={<Analytics />} />
            <Route path="system-health" element={<SystemHealth />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
