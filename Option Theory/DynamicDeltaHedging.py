
        
@dataclass(frozen=True)
class HedgingConfig:
    S0: float
    K: float
    T: float                 # years
    r: float                 # annual (cont.)
    sigma: float             # annual vol used for BOTH sim and delta
    total_shorted: float     # cash raised at t=0 from short call (e.g., priced with σ_real or σ_implied)
    M: int                   # number of paths
    n_contracts: int
    contract_size: int
    n_steps: int             # number of re-hedging steps
    mu: float
    option_type: str         # "call" or "put"
    position: str                 # real-world drift (scalar)
    seed: int = 42 
    bs: BlackScholesPricer


class DynamicHedgingRunner:

    def __init__(self, params: HedgingConfig):
        self.params = params
        
    def black_scholes_delta(self,S, K, T, r, sigma,option_type):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        if option_type == "call":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
        
    

    def dynamic_hedging(self):
        
                dt = self.params.T / self.params.n_steps  
                S_paths = np.zeros((self.params.M, self.params.n_steps + 1)) 
                S_paths[:, 0] = self.params.S0  
                
                # Simulate stock price paths under real-world measure
                for t in range(1, self.params.n_steps + 1):
                    Z = np.random.standard_normal(self.params.M)  # Standard normal random variables
                    S_paths[:, t] = S_paths[:, t-1] * np.exp((self.params.mu - 0.5 * self.params.sigma ** 2) * dt + self.params.sigma * np.sqrt(dt) * Z)

                
                position_sign = 1 if self.params.position.lower() == "short" else -1


                # Initial delta is equal to the Black Scholes model N(d1)
                delta = self.black_scholes_delta(self.params.S0, self.params.K, self.params.T, self.params.r, self.params.sigma,self.params.option_type)[0]

                hedge_ratio = - position_sign * delta
                stock_holdings = np.full(self.params.M, hedge_ratio)

                # Initialize the value of the initial stock holding
                initial_stock_holding = hedge_ratio * self.params.n_contracts * self.params.contract_size  # scaled shares (for cash only)
                price = self.params.bs(BSParams(r = self.params.r, sigma = self.params.T, S0 = self.params.S0, K = self.params.K)).call() if self.params.option_type == "call" else self.params.bs(BSParams(r = self.params.r, sigma = self.params.T, S0 = self.params.S0, K = self.params.K)).put()
                premium = price * self.params.n_contracts * self.params.contract_size

                total_shorted = position_sign * premium


                # Initial cash after buying long position
                cash_account = np.full(self.params.M,total_shorted) - initial_stock_holding * self.params.S0

                # Rebalance portfolio
                for t in range(1, self.params.n_steps + 1):
                    T_t = self.params.T - t * dt  
                    cash_account *= np.exp(self.params.r * dt)
                    
                            
                    # Update delta for the new time and new stock price
                    if t < self.params.n_steps:
                        delta_new = self.black_scholes_delta(S_paths[:, t], self.params.K, T_t, self.params.r, self.params.sigma,self.params.option_type)
                    else:
                        # last step: T_t == 0 → use indicator
                        delta_new = (S_paths[:, t] > self.params.K).astype(float) if self.params.option_type == "call" else -(S_paths[:, t] < self.params.K).astype(float)

                    new_hedge_ratio = -position_sign * delta_new
                    delta_change = new_hedge_ratio - stock_holdings
                    cash_account -= delta_change * S_paths[:, t] * self.params.n_contracts * self.params.contract_size
                    stock_holdings = new_hedge_ratio

                # Final portfolio value at maturity
                
                # Final portfolio value at maturity
                final_stock_value = stock_holdings * S_paths[:, -1] * self.params.n_contracts * self.params.contract_size
                final_hedge_value = cash_account + final_stock_value

                        
                # Payoff magnitude of the option (holder's payoff)
                if self.params.option_type == "call":
                    payoff = np.maximum(S_paths[:, -1] - self.params.K, 0.0)
                else:
                    payoff  = np.maximum(self.params.K - S_paths[:, -1], 0.0)
                payoff *= self.params.n_contracts * self.params.contract_size

                # Apply position sign: short => subtract payoff; long => add payoff
                payoff_sign = - position_sign
                PnL = final_hedge_value + payoff_sign * payoff
                        
                # Print results
                mean = float(PnL.mean())
                std = float(PnL.std(ddof=1))
                p5, p50, p95 = [float(np.percentile(PnL, q)) for q in (5, 50, 95)]
                pmin, pmax = float(PnL.min()), float(PnL.max())

                print(f"[Dynamic Hedging Result]")
                print(f"n_steps={self.params.n_steps}, mu={self.params.mu}")
                print(f"PnL summary: mean={mean:.6f}, std={std:.6f}, "f"p5={p5:.6f}, median={p50:.6f}, p95={p95:.6f}, "f"min={pmin:.6f}, max={pmax:.6f}")

                return PnL
    