import math
import pandas
import yfinance


class CustomIndicators:

    # WILLIAMS VIX FIX WITH BB AND STD

    def get_wvf(self, source, length, lows):

        WVFs = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None]

        working_closes = []
        count = 0
        highest_closes = []
        lowest_closes = []
        for close in source:
            count += 1
            working_closes.append(close)
            if count > (length - 1):
                highest_closes.append(max(working_closes))
                lowest_closes.append(min(working_closes))
                working_closes.remove(working_closes[0])
                count -= 1

        for i in range(len(highest_closes)):
            WVFs.append(((highest_closes[i] - lows[i+21]) / highest_closes[i]) * 100)

        return WVFs

    def get_sma(self, source, length):

        SMAs = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None]

        SUM = 0
        the_closes = []
        count = 0
        for close in source:
            if close is not None:
                count += 1
                the_closes.append(close)
                if count > (length - 1):
                    for i in the_closes:
                        SUM += (i / length)
                    count -= 1
                    SMAs.append(SUM)
                    the_closes.remove(the_closes[0])
                    SUM = 0

        return SMAs

    def get_sum(self, fst, snd):

        EPS = 1e-10
        res = fst + snd
        if abs(res) <= EPS:
            res = 0
        else:
            if not abs(res) <= 1e-4:
                res = res
            else:
                res = 15

        return res

    def sma_std(self, source, length):

        STDs = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None]

        SMAs = self.get_sma(source=source, length=length)
        sum_of_square_deviations = 0
        count = 0
        the_sums = []
        for i in range(len(SMAs)):
            if SMAs[i] is not None:
                count += 1
                the_sum = self.get_sum(fst=source[i], snd=-SMAs[i])
                the_sums.append(the_sum)
                if count > (length - 1):
                    for sum in the_sums:
                        sum_of_square_deviations += (sum * sum)
                    count -= 1
                    stdev = math.sqrt(sum_of_square_deviations / length)
                    STDs.append(stdev)
                    the_sums.remove(the_sums[0])
                    sum_of_square_deviations = 0

        return {'STDs': STDs, 'SMAs': SMAs}

    def mult_sdev(self, source, mult):

        new_STDs = []
        for i in source.get('STDs'):
            if i is None:
                new_STDs.append(None)
            else:
                new_STDs.append(i * mult)

        return new_STDs

    def BB_bands(self, SMAs, STDs):

        lower = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                 None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                 None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                 None, None]
        upper = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                 None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                 None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                 None, None]
        for idx, i in enumerate(STDs):
            if STDs[idx] is not None:
                lower.append(SMAs.get('SMAs')[idx] - STDs[idx])
                upper.append(SMAs.get('SMAs')[idx] + STDs[idx])

        return {'low BB': lower, 'up BB': upper}

    def get_high_and_low(self, source, length, high_pe, low_pe):

        range_high = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                      None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                      None, None,
                      None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                      None,
                      None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                      None, None, None]
        range_low = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                     None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                     None, None,
                     None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                     None,
                     None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                     None, None, None]

        working_closes = []
        count = 0
        for wvf in source:
            if wvf is not None:
                count += 1
                working_closes.append(wvf)
                if count > (length - 1):
                    range_high.append(max(working_closes) * high_pe)
                    range_low.append(min(working_closes) * low_pe)
                    working_closes.remove(working_closes[0])
                    count -= 1

        return {'high ranges': range_high, 'low ranges': range_low}


    def Williams_VIX_Fix_BB_STD(self, closes):

        stock_pd = closes
        add_pd = {'WVF': None, 'STD': None, 'BB Up': None, 'BB Low': None, 'WVF SMA': None, 'High Range': None, 'Low Range': None}

        WVF_lookback_p_STD = 22
        BB_length = 20
        BB_STD_up = 2
        lookback_p_high_pe = 50
        highest_pe = .85
        lowest_pe = 1.01

        WVFs = self.get_wvf(source=stock_pd['Close'].to_list(), length=WVF_lookback_p_STD, lows=stock_pd['Low'].to_list())
        add_pd['WVF'] = WVFs

        STDs_and_SMAs = self.sma_std(source=WVFs, length=BB_length)
        add_pd['WVF SMA'] = STDs_and_SMAs.get('SMAs')

        mult_stdev = self.mult_sdev(source=STDs_and_SMAs, mult=BB_STD_up)
        add_pd['STD'] = mult_stdev

        low_and_up_bands = self.BB_bands(SMAs=STDs_and_SMAs, STDs=mult_stdev)
        add_pd['BB Up'] = low_and_up_bands.get('up BB')
        add_pd['BB Low'] = low_and_up_bands.get('low BB')

        high_and_low_range = self.get_high_and_low(source=WVFs, length=lookback_p_high_pe, high_pe=highest_pe,
                                                   low_pe=lowest_pe)
        add_pd['High Range'] = high_and_low_range.get('high ranges')
        add_pd['Low Range'] = high_and_low_range.get('low ranges')

        new_df = pandas.DataFrame(add_pd, index=stock_pd.index)
        stock_pd = stock_pd.join(new_df)

        return stock_pd

    # SUPERTREND

    def tr(self, data):
        data['previous_close'] = data['Close'].shift(1)
        data['high-low'] = abs(data['High'] - data['Low'])
        data['high-pc'] = abs(data['High'] - data['previous_close'])
        data['low-pc'] = abs(data['Low'] - data['previous_close'])

        tr = data[['high-low', 'high-pc', 'low-pc']].max(axis=1)

        return tr

    def atr(self, data, period):
        data['tr'] = self.tr(data)
        atr = data['tr'].rolling(period).mean()

        return atr

    def supertrend(self, df, period=10, atr_multiplier=3):
        hl2 = (df['High'] + df['Low']) / 2
        df['atr'] = self.atr(df, period)
        df['upperband'] = hl2 + (atr_multiplier * df['atr'])
        df['lowerband'] = hl2 - (atr_multiplier * df['atr'])
        df['in_uptrend'] = True

        for current in range(1, len(df.index)):
            previous = current - 1

            if df['Close'][current] > df['upperband'][previous]:
                df['in_uptrend'][current] = True
            elif df['Close'][current] < df['lowerband'][previous]:
                df['in_uptrend'][current] = False
            else:
                df['in_uptrend'][current] = df['in_uptrend'][previous]

                if df['in_uptrend'][current] and df['lowerband'][current] < df['lowerband'][previous]:
                    df['lowerband'][current] = df['lowerband'][previous]

                if not df['in_uptrend'][current] and df['upperband'][current] > df['upperband'][previous]:
                    df['upperband'][current] = df['upperband'][previous]

        return df

    # BETA

    def covariance(self, SPY_hist, hist2):

        SPY_closes = SPY_hist
        length = len(SPY_closes)

        SPY_average = 0
        for p in SPY_closes:
            SPY_average += p
        SPY_average = SPY_average / length

        hist2_closes = hist2
        length = len(hist2_closes)

        hist2_average = 0
        for day in hist2_closes:
            hist2_average += day
        hist2_average = hist2_average / length

        total = 0
        for i in range(len(SPY_hist)):
            total += ((SPY_closes[i] - SPY_average) * (hist2_closes[i] - hist2_average))

        return total / (length - 1)

    def variance(self, SPY_hist):

        SPY_closes = SPY_hist
        length = len(SPY_closes)

        SPY_average = 0
        for p in SPY_closes:
            SPY_average += p
        SPY_average = SPY_average / length

        sum = 0
        for i in SPY_closes:
            dif = i - SPY_average
            sum += (dif ** 2)

        return (sum / (length-1))


    def beta(self, SPY_hist, hist2):

        _covariance = self.covariance(SPY_hist=SPY_hist, hist2=hist2)

        _variance = self.variance(SPY_hist=SPY_hist)

        return (_covariance / _variance)

    def percent_change(self, hist):

        closes = hist['Close'].to_list()
        percent_change = []
        for i in range(len(closes)):
            if i != 0:
                percent_change.append(((closes[i-1] - closes[i]) / closes[i-1]) * 100)

        return percent_change

    def get_beta(self, securities):

        SPY_hist = yfinance.Ticker('SPY').history(period='5y', interval='1mo')  # using this beta for now because it seems to be standard
        SPY_pc = self.percent_change(hist=SPY_hist)
        betas = {}

        for security in securities:
            security_hist = yfinance.Ticker(security).history(period='5y', interval='1mo')
            security_pc = self.percent_change(hist=security_hist)
            b = self.beta(SPY_hist=SPY_pc, hist2=security_pc)
            betas[security] = b

        return betas

