import data
import Training
import QCNN_circuit
import numpy as np
import pennylane as qml
import MC_QCNN_circuit
# imports
import math


def accuracy_test(predictions, labels, cost_fn, binary = True):
    if cost_fn == 'mse':
        if binary == True:
            acc = 0
            for l, p in zip(labels, predictions):
                if np.abs(l - p) < 1:
                    acc = acc + 1
            return acc / len(labels)

        else:
            acc = 0
            for l, p in zip(labels, predictions):
                if np.abs(l - p) < 0.5:
                    acc = acc + 1
            return acc / len(labels)

    elif cost_fn == 'cross_entropy':
        acc = 0
        for l,p in zip(labels, predictions):
            if p[0] > p[1]:
                P = 0
            else:
                P = 1
            if P == l:
                acc = acc + 1
        return acc / len(labels)


def Encoding_to_Embedding(Encoding):
    # Amplitude Embedding / Angle Embedding
    if Encoding == 'resize256':
        Embedding = 'Amplitude'
    elif Encoding == 'pca8':
        Embedding = 'Angle'
    elif Encoding == 'autoencoder8':
        Embedding = 'Angle'

    # Amplitude Hybrid Embedding
    # 4 qubit block
    elif Encoding == 'pca32-1':
        Embedding = 'Amplitude-Hybrid4-1'
    elif Encoding == 'autoencoder32-1':
        Embedding = 'Amplitude-Hybrid4-1'

    elif Encoding == 'pca32-2':
        Embedding = 'Amplitude-Hybrid4-2'
    elif Encoding == 'autoencoder32-2':
        Embedding = 'Amplitude-Hybrid4-2'

    elif Encoding == 'pca32-3':
        Embedding = 'Amplitude-Hybrid4-3'
    elif Encoding == 'autoencoder32-3':
        Embedding = 'Amplitude-Hybrid4-3'

    elif Encoding == 'pca32-4':
        Embedding = 'Amplitude-Hybrid4-4'
    elif Encoding == 'autoencoder32-4':
        Embedding = 'Amplitude-Hybrid4-4'

    # 2 qubit block
    elif Encoding == 'pca16-1':
        Embedding = 'Amplitude-Hybrid2-1'
    elif Encoding == 'autoencoder16-1':
        Embedding = 'Amplitude-Hybrid2-1'

    elif Encoding == 'pca16-2':
        Embedding = 'Amplitude-Hybrid2-2'
    elif Encoding == 'autoencoder16-2':
        Embedding = 'Amplitude-Hybrid2-2'

    elif Encoding == 'pca16-3':
        Embedding = 'Amplitude-Hybrid2-3'
    elif Encoding == 'autoencoder16-3':
        Embedding = 'Amplitude-Hybrid2-3'

    elif Encoding == 'pca16-4':
        Embedding = 'Amplitude-Hybrid2-4'
    elif Encoding == 'autoencoder16-4':
        Embedding = 'Amplitude-Hybrid2-4'

    # Angular HybridEmbedding
    # 4 qubit block
    elif Encoding == 'pca30-1':
        Embedding = 'Angular-Hybrid4-1'
    elif Encoding == 'autoencoder30-1':
        Embedding = 'Angular-Hybrid4-1'

    elif Encoding == 'pca30-2':
        Embedding = 'Angular-Hybrid4-2'
    elif Encoding == 'autoencoder30-2':
        Embedding = 'Angular-Hybrid4-2'

    elif Encoding == 'pca30-3':
        Embedding = 'Angular-Hybrid4-3'
    elif Encoding == 'autoencoder30-3':
        Embedding = 'Angular-Hybrid4-3'

    elif Encoding == 'pca30-4':
        Embedding = 'Angular-Hybrid4-4'
    elif Encoding == 'autoencoder30-4':
        Embedding = 'Angular-Hybrid4-4'

    # 2 qubit block
    elif Encoding == 'pca12-1':
        Embedding = 'Angular-Hybrid2-1'
    elif Encoding == 'autoencoder12-1':
        Embedding = 'Angular-Hybrid2-1'

    elif Encoding == 'pca12-2':
        Embedding = 'Angular-Hybrid2-2'
    elif Encoding == 'autoencoder12-2':
        Embedding = 'Angular-Hybrid2-2'

    elif Encoding == 'pca12-3':
        Embedding = 'Angular-Hybrid2-3'
    elif Encoding == 'autoencoder12-3':
        Embedding = 'Angular-Hybrid2-3'

    elif Encoding == 'pca12-4':
        Embedding = 'Angular-Hybrid2-4'
    elif Encoding == 'autoencoder12-4':
        Embedding = 'Angular-Hybrid2-4'

    # Two Gates Compact Encoding
    elif Encoding == 'pca16-compact':
        Embedding = 'Angle-compact'
    elif Encoding == 'autoencoder16-compact':
        Embedding = 'Angle-compact'
        
    else:
        print("Error: Wrong Encoding Input")
        
    return Embedding


def Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings, circuit, cost_fn, binary=True):
    I = len(Unitaries)
    J = len(Encodings)

    for i in range(I):
        for j in range(J):
            f = open(r'C:\Users\charu\Desktop\Projects\MNIST_QCNN\Result_BRATS\result.txt', 'a')
            f1 = open(r'C:\Users\charu\Desktop\Projects\MNIST_QCNN\Result_BRATS\dropout.txt','a')
            f.write("---------\n")
            f1.write("----------\n")
            U = Unitaries[i]
            U_params = U_num_params[i]
            Encoding = Encodings[j]
            Embedding = Encoding_to_Embedding(Encoding)
            print("Generating data now...")
            X_train, X_test, Y_train, Y_test, X_val, Y_val = data.data_load_and_process(dataset, classes=classes,
                                                                          feature_reduction=Encoding, binary=binary)
            print("Data generated")
            
            print("\n")
            print("Loss History for " + circuit + " circuits, " + U + " " + Encoding + " with " + cost_fn)
            print("training now")
            #TRAINING
            loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, Embedding, circuit, cost_fn)
        
            # loss_history = [0.3562709141264657, 0.2586827702091999, 0.25571488373134266, 0.24829546615991513, 0.1491659924490176, 0.29614337947448405, 0.17622072805088612, 0.2106475700236631, 0.2783999337628257, 0.19814617345594596, 0.2123485072545478, 0.20965527296147657, 0.2030790186818402, 0.2179428294058778, 0.1990548139620231, 0.19397799473684615, 0.2054138144047146, 0.20635659746693658, 0.18824822472334685, 0.22081272863261545, 0.18201161776613534, 0.16611868561049914, 0.22584565473137574, 0.17893724994244775, 0.22566303367489465, 0.1396213039983739, 0.18140327319000327, 0.21918107690169192, 0.188717687843236, 0.1522158855233644, 0.20999980226702586, 0.20717402278395333, 0.18880914295533546, 0.21837743361004103, 0.16267097893816063, 0.22077967617114758, 0.1387742530522562, 0.13832645835819676, 0.20355690849341698, 0.19845836993439955, 0.16377651479384472, 0.19538870571672193, 0.2027726350834685, 0.19538952082309197, 0.1715593142752493, 0.20592158839424302, 0.18644009120312455, 0.14810178760858972, 0.1877253008996474, 0.18782724130555634, 0.19527341995528955, 0.21401145849304842, 0.18079174921871322, 0.15628367789424188, 0.17343498858654607, 0.16534966953658883, 0.1869435562071061, 0.15366911553796261, 0.1789629035617383, 0.1617913494545264, 0.1356300148475859, 0.2088869257189495, 0.12577813801682486, 0.2327344177997078, 0.20974039716540122, 0.17935656738307357, 0.1701796679382278, 0.19677937957143135, 0.19632320317695837, 0.2136319965973616, 0.19853121306886515, 0.1624949048533688, 0.19428392690855875, 0.2025181772402342, 0.14873824234367589, 0.20350407583114402, 0.1806177559560054, 0.14917477337395194, 0.15284455753223547, 0.16387393352782567, 0.146079497828391, 0.23657224220085213, 0.22891254495193686, 0.15429017349355498, 0.20573122220689807, 0.14611109191453006, 0.1367969075270774, 0.20480060661297064, 0.17124450939400251, 0.17892435501624912, 0.23409277239643453, 0.1707785521334984, 0.18996175417339636, 0.1372989351835386, 0.16205625080873282, 0.1943396973074341, 0.17949771552141194, 0.17879382680427616, 0.179875340778561, 0.17556373995490984, 0.15367839667717534, 0.1955579672848352, 0.17831424164225815, 0.20536390458360013, 0.1706156618007271, 0.1712767779242163, 0.23791705856035072, 0.19083128067475758, 0.17116591607021467, 0.23412458359250113, 0.2017573353758498, 0.1652012523234734, 0.20236872608862141, 0.20574647182044298, 0.15237654159169248, 0.23115017452668724, 0.18022637765223642, 0.1654884222611443, 0.2022163905624979, 0.23102301221613275, 0.13790842425772115, 0.17479248336809428, 0.17190000574482622, 0.21069975044915776, 0.16235495324395074, 0.1548706069589474, 0.21916461434604115, 0.21157636998609386, 0.16971833777184003, 0.17977631723416146, 0.1539410421437172, 0.2238230946165756, 0.19002992652852588, 0.169613378355713, 0.16194030518376637, 0.14608337387809464, 0.17666491092689857, 0.15324819859633912, 0.15252957959729385, 0.17014743648884934, 0.14299719745724654, 0.1891630803849973, 0.15200128543131747, 0.2167826678490052, 0.14924928676090427, 0.17400682432544856, 0.20214211453362702, 0.1799426063487135, 0.17858429605133241, 0.16139561297658703, 0.17887848537138185, 0.1351119835254504, 0.1529698287619231, 0.17957872507175063, 0.18873975020261105, 0.13836905675180597, 0.12636217560201007, 0.1966953181314488, 0.16965504588166475, 0.2355730296921945, 0.14411949012554162, 0.1700587839492402, 0.222724601239986, 0.1526441093927727, 0.1793504701851668, 0.16981162974611003, 0.1542993851256285, 0.2044565755722855, 0.17096118235934513, 0.17882234667900898, 0.20422132586656866, 0.20308812127308068, 0.15490669065124554, 0.12278001815385414, 0.13754551474855375, 0.17790764258430053, 0.16168933463300644, 0.15314269663858804, 0.14236116616600392, 0.14223247254905885, 0.16178447951785035, 0.123214797997538, 0.1625130499539076, 0.10159402081093968, 0.1720326064088435, 0.18435652249071482, 0.18367565762839344, 0.14267372899074734, 0.11093211363725933, 0.18917082723790843, 0.19405925428002982, 0.19218985605811179, 0.18091428582062033, 0.200506020166842, 0.1430250095890581, 0.14406900810308534, 0.17014879341501132, 0.18792227721621807, 0.1907579878767903, 0.16164161179625333, 0.18320265604860483, 0.15611158294667965, 0.14537707277352294, 0.2290621506234249, 0.13840979721463978, 0.16151467985727097, 0.22991626791253744, 0.15432973805067732, 0.17026153560809323, 0.195442484544959, 0.17842862521339545, 0.19590984517475724, 0.20355480456659944, 0.17084106513138186, 0.1459083344396178, 0.19449442395498337, 0.16234557359008903, 0.16861165526758595, 0.10484231186085656, 0.14511593832187789, 0.1967943400040772, 0.169327675575755, 0.15273674372660184, 0.1891361960347888, 0.1233168099922473, 0.2142435320001369, 0.20403719193347192, 0.1337540515325879, 0.1705261886446101, 0.18331668545166194, 0.1430034665974534, 0.16101952596895158, 0.1429033172400518, 0.17977979924841891, 0.17349305144630578, 0.18887867199482955, 0.19728570313346017, 0.1167136610803546, 0.1709972110647618, 0.2067360292911349, 0.1820238107370778, 0.16975195773321594, 0.15392780554415592, 0.1522877487569584, 0.20359177876447865, 0.21271511554834804, 0.1664604153257058, 0.12067709823743554, 0.1697944558567508, 0.2258235647315424, 0.19838870543332776, 0.18809348155575015, 0.2036688699919967, 0.195255897558391, 0.16897318162651237, 0.19430633494599717, 0.16395666374428586, 0.1312761989252043, 0.19854141738515502, 0.2342430426138823, 0.1668910769174585, 0.2111811320412834, 0.1476774072633359, 0.1867104470238096, 0.1877451891172279, 0.14734896707729533, 0.21906848034686768, 0.16241838634796926, 0.14693098436457122, 0.13702493068017602, 0.1559158199063108, 0.21321278882051617, 0.20567502143597796, 0.16084439525520505, 0.16113208784181865, 0.17870115735425154, 0.1898648145294872, 0.16143750119168268, 0.13566421093955774, 0.15133815163401534, 0.15470344558675228, 0.1665034279122141, 0.15281955285598778, 0.2081318542550344, 0.1523512390522541, 0.18043879897725956, 0.11409682196737163, 0.16950391793285258, 0.13277449263969396, 0.1222243918157799, 0.20190518138423244, 0.16094004321789165, 0.1521854340761393, 0.15204050819415152, 0.12218636197750497, 0.16114733765627384, 0.14281957415058716, 0.16149628659783008, 0.21199704761942217, 0.19045894396276358, 0.1715484703383776, 0.1821576441757867, 0.17000174979680072, 0.2060191691733547, 0.2138026231600088, 0.1540896128150653, 0.17918382525246798, 0.22142251052259504, 0.1901895568228185, 0.15716040674885276, 0.16406656808098352, 0.16393300999992433, 0.17116712280275104, 0.17943738374071747, 0.18693633440959215, 0.13953165551378038, 0.17965254582597628, 0.17528419372296808, 0.1638006440798129, 0.13556159664278553, 0.16066746295061693, 0.1618677753297722, 0.16206769574692026, 0.1696979720886943, 0.1507198107581448, 0.1502751987048795, 0.17218496105630887, 0.17129737430262384, 0.15135022641950854, 0.18171541697382845, 0.19083519139432203, 0.15179854173708549, 0.2092651023969147, 0.16987314678357718, 0.20753218894062575, 0.1706931520738378, 0.1449374032717472, 0.17067963603178388, 0.17086470207721882, 0.16134773462696686, 0.13704801203689043, 0.12190665433402242, 0.17831186173765473, 0.23128803276739243, 0.16961244605502745, 0.11809254896259908, 0.1552903213559545, 0.1984904693610014, 0.17027288189548537, 0.16964805719204096, 0.17959330503654117, 0.2148530796088966, 0.14303080674273483, 0.17885763782169367, 0.21312434609180925, 0.180484637221447, 0.20420398427057246, 0.14086761789919333, 0.17890946990637324, 0.19479401309517885, 0.1788894682056089, 0.2268508432946711, 0.15603677916259012, 0.14050916547777767, 0.13859122732839338, 0.15330587659381617, 0.11921589527960003, 0.13449845727873538, 0.1718711717646283, 0.17995814465475016, 0.1614615038364235, 0.16127263660096267, 0.17386468863888746, 0.14254212719051493, 0.13243691179234116, 0.1829782566579007, 0.18098806308570528, 0.20260785954180385, 0.19149473684171045, 0.15199981105245536, 0.2092271759290129, 0.18830453467093267, 0.2234749874750346, 0.18928907241664888, 0.1858513808139737, 0.18931671540326994, 0.1340361931106217, 0.19476333465770876, 0.1893989084941725, 0.21557683405162173, 0.1644077701954126, 0.17977724489780209, 0.16549543657764418, 0.22419655506607267, 0.23145593490356955, 0.15891044808789206, 0.1804672246022187, 0.13480745478290376, 0.20369378432880006, 0.1317151712575473, 0.13787086453062672, 0.13572254469092934, 0.17849907644297744, 0.20721890549560504, 0.14174435882049585, 0.16219393765307233, 0.15193380843003151, 0.10255692140408275, 0.1899010812459196, 0.19290347997560028, 0.15054709500867353, 0.17354271447450437, 0.1926416071200823, 0.20616075517792104, 0.19066063711438538, 0.13297092321324142, 0.16059094378421984, 0.15535089379348482, 0.17880654023965054, 0.16966993617971057, 0.15201056691274997, 0.18777574020901666, 0.11813817606676759, 0.1523481844774644, 0.13485832196608596, 0.1887793714855159, 0.17849804209950573, 0.16181326724531916, 0.1881351901106327, 0.21607594877811134, 0.1794172314607766, 0.12674052194348717, 0.1880044625053938, 0.1964009200457575, 0.13628198365011832, 0.17015958975847895, 0.16396114990582902, 0.16915209642125, 0.16800643185214983, 0.1970719730939294, 0.13591498525983348, 0.2051448965935626, 0.1701712435207151, 0.17962637961332034, 0.15187341074165228, 0.18248213301138774, 0.21839833739255463, 0.19161255376022118, 0.18718272769966432, 0.17750685218451157, 0.1933040872641142, 0.14751840846740533, 0.14603985433872124, 0.1779888210069785, 0.1879283330586129, 0.19497882038291475, 0.17041325917900713, 0.17084216520943624, 0.15296937131965138, 0.18893911937888247, 0.22995164207635663, 0.2364945012927575, 0.21016648318797038, 0.17841595358660564, 0.15495236841773982, 0.1816131185142217, 0.21736076325177073, 0.15855214888167563, 0.11976518593754737, 0.14752112159489397, 0.22152252740850542, 0.1485999077990646, 0.17783620218899632, 0.1780445871501345, 0.18007402763425928, 0.17838915332964747, 0.15091324489579389, 0.18583476187126355, 0.16134306234246193, 0.16673888369222165, 0.18089975189997784, 0.2343415880425519, 0.1708382052380405, 0.17743083838814283, 0.18599583993562957, 0.1528884995843261, 0.1538403848974954, 0.1786211985139415, 0.17139557931846666, 0.19613516063168612, 0.16926510673785308, 0.18300346732377132, 0.19728002037755527, 0.15311044007660132, 0.1462407170484213, 0.20593723762559005, 0.15297156874438644, 0.21982829737854345, 0.1537759614862446, 0.14529911036142792, 0.13789083220969475, 0.13539052653087488, 0.1436659435023886, 0.21709604303534485, 0.1520426491903496, 0.18950425310918123, 0.1614107316993904, 0.1704620488947349, 0.15051540638710909, 0.16915081125221146, 0.18700923618742207, 0.18091813778049332, 0.1521520347783044, 0.19533444689943594, 0.1702051531148059, 0.1774543234972475, 0.1260601591070921, 0.22206046624578096, 0.1976657845991296, 0.13770628489699002, 0.15367440429018533, 0.21451466705509697, 0.21191879281375497, 0.18686626073519144, 0.14587688658269607, 0.16641647799368092, 0.18494551325544253, 0.23378232152420583, 0.17701522565554018, 0.20331861780576932, 0.16399670576111872, 0.19503390285433558, 0.20022552000605387, 0.14255849570740098, 0.19544363103346074, 0.21058327382695843, 0.1632170916245557, 0.14031245732935663, 0.1721079292748115, 0.16259500388190595, 0.14616849696485212, 0.17475380093161652, 0.21533434560578169, 0.1441008126776412, 0.19727083056034456, 0.15287847272073776, 0.1888198095473979, 0.197975904903042, 0.1804970509066565, 0.23284537843087924, 0.2028066420423521, 0.18717363695220987, 0.17177353426006342, 0.15306161320139677, 0.18188472371132583, 0.20242235141782097, 0.17012859126033827, 0.13413817766233688, 0.17138808777819592, 0.18644357736187267, 0.13757056758847291, 0.17014295121313655, 0.2124529906488334, 0.17081114765614816, 0.14337713174058603, 0.16164154683968437, 0.1591426996311025, 0.18004738209538718, 0.15890652659141932, 0.19070710185373105, 0.18881397941054126, 0.20689148988210515, 0.16866576663610622, 0.19522297687813317, 0.1621224545083856, 0.20429698800293866, 0.20940102348244344, 0.130766083280773, 0.1570558062857855, 0.18578792952220693, 0.1440531855916408, 0.19576941316235802, 0.16849913442439612, 0.14349396647843246, 0.1528948241575077, 0.15243225434968677, 0.18942041687838132, 0.17005141504391655, 0.22163700718743692, 0.17767607191863782, 0.12602539103643418, 0.22514353407947463, 0.23277911423216024, 0.19944865205486742, 0.18761537354408678, 0.1942468157187345, 0.14723677897042328, 0.162062249623461, 0.16924332924026275, 0.19424687145893177, 0.16892341358398807, 0.16973181548543284, 0.1933473066287936, 0.16816477433587493, 0.1388545980534248, 0.13736691340011078, 0.1797245653960954, 0.2042534365678377, 0.22344143943039563, 0.17875780618739684, 0.15190391585895682, 0.184631151497408, 0.10838322076043477, 0.16007436960676602, 0.17964822381430792, 0.18822668941878615, 0.1767128285224091, 0.2055308911153099, 0.1345313076646092, 0.2053660176714054, 0.16090828956899395, 0.15251561505423997, 0.17858366632921005, 0.17876007656318488, 0.24632997980825524, 0.21087015559441308, 0.15260993683176355, 0.18603571289541007, 0.186249639323791, 0.18722101850669248, 0.20740891662882546, 0.2086340687157696, 0.19410148330342794, 0.18726301851967253, 0.20696146156005962, 0.18184678501942775, 0.2085943832180886, 0.18297967790786274, 0.20334097574116003, 0.1822035972380133, 0.214843370195306, 0.15918081811248155, 0.174220795592646, 0.14938652100827463, 0.17870827160281524, 0.1547366861711206, 0.15381456908935182, 0.2045311767390137, 0.22267208647336909, 0.18649101377655836]

            # trained_params = [-0.44741395,  0.5582139,  -1.54813476, -0.34688524,  0.63349681, -0.09947869,
            # 0.28609961,  0.76814148, 1.20909747 , 0.53075586, -1.41232621, -1.04779804,
            # 1.61994592, -0.44590387, -0.92657193,  0.6547805,   0.00309325, -1.43122777,
            # -0.32449462,  0.27198128,  0.74842264, -0.96735965,  1.46371556, -0.63105987,
            # 1.01936219,  1.30344929,  0.44590104, -0.68906581, -2.250847,   -0.20201772,
            # 0.72459041,  1.01627888, -1.08071021, -0.82182626 , 0.36659268, -0.75632439,
            # -0.44281503,  1.03783783, -1.52751822,  1.08590089 , 1.13995789, -0.20577166,
            # 0.36839681, -1.89287473,  0.19218937,  0.26011822,-0.22653156, -0.2407393,
            # 0.97645158,  1.24410275,  2.68237655,  0.00754572, -0.61932187,  0.65155668,
            # -0.14506046, -1.14149104,  0.19377894, -1.59754824, -1.23459931,  0.96784343,
            # -0.68260724 , 0.72709306,  0.09878107]
            trained_params = [-1.13, 0.0, -0.74, -1.41, 0.73, -1.39,
            0.90, 0.0, -1.61, -1.49, 0.94, -0.90,
            0.0, 0.0, 1.87, 0.68, -1.97, 0.0,
            0.0, 1.13, 1.24, -1.66, 1.20, -0.99,
            1.64, 0.0, 1.87, 0.81, 0.68, -0.57,
            0.0, 1.21, -1.72, -1.42, -0.74, 0.87,
            -1.53, 1.39, -0.89, 1.52, 1.74, -1.66,
            -1.99, 0.0, 0.0, -1.11, -0.60, 1.89,
            -1.39, -1.13, 0.0, 1.39, -1.71, 0.97,
            0.83, 1.65, 0.0, -0.66, 1.10, -0.72,
            0.62, -0.69, -1.34, -1.08, -0.75, -1.45,
            0.95, 0.89, -0.91]


            # print(trained_params)

            # trained_2 = [0.0 for i in range(len(trained_params))]
            
            # print("Len of trained_params: " + trained_params)
            # num_params = 0
            # # after training, make a few params close to zero
            # for idx in range(len(trained_params)):
            #     if math.isclose(trained_params[idx], 0.0, abs_tol=0.3):
            #         print("Parameter: " , trained_params[idx])
            #         num_params += 1
            #         trained_params[idx] = 0.0
            # print ("Number of parameters changed: " + num_params)
            # f.write("Loss History for " + circuit + " circuits, " + U + " " + Encoding + " with " + cost_fn)
            # f.write("\n")
            # f.write(str(loss_history))
            #TESTING
            predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding, cost_fn) for x in X_test]
            # predictions = [QCNN_circuit.QCNN(x, trained_2, U, U_params, Embedding, cost_fn) for x in X_test]
            accuracy = accuracy_test(predictions, Y_test, cost_fn, binary)
            print("Accuracy for testing " + U + " " + Encoding + " :" + str(accuracy))

            
            f.write("\n") 
            f.write("Accuracy for testing " + U + " " + Encoding + " :" + str(accuracy))
            f.write("\n")
            f.write("\n")
            
            #VALIDATION
            predictions_val = [QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding, cost_fn) for x in X_val]
            # predictions_val = [QCNN_circuit.QCNN(x, trained_2, U, U_params, Embedding, cost_fn) for x in X_val]
            #print(qml.draw(QCNN_circuit.QCNN)(X_test[0], trained_params, U, U_params, Embedding, cost_fn))
            #add testing iterations with monte carlo for which create circuits with same parameters and drop gates.
            
            accuracy = accuracy_test(predictions_val, Y_val, cost_fn, binary)
            print("Accuracy for validation " + U + " " + Encoding + " :" + str(accuracy))

            # f.write("Loss History for " + circuit + " circuits, " + U + " " + Encoding + " with " + cost_fn)
            # f.write("\n")
            # f.write(str(loss_history))
            f.write("\n")
            f.write("Accuracy for validation " + U + " " + Encoding + " :" + str(accuracy))
            f.write("\n")
            f.write("\n")
            
            #MC DROPOUT TESTING AND VALIDATION
            '''parameters to send: import mcdropout get QCNN
                trained_params, U, U_paramas, percentage of dropout2
                this will return result i.e. prediction 
            
                values needed in return :predictions for all possible dropouts in each layers
                need to make loop for getting acc of vals 
                of another accuracy checking algo that returns average testing and validation value after dropouts
                
            
            #trained_params = [0.7506303298494379, 0.7395528838823279, 0.5321771317300852, 0.39453322091049003, 0.31735972975628823, 0.3092162762712195, 0.28379452775541625, 0.28894306968195393, 0.2661403915084277, 0.2905249354684899, 0.2730646174529658, 0.27663046859001544, 0.25524669801708166, 0.24703400066250328, 0.2200199175965394, 0.26270766900398196, 0.2446219856752722, 0.25728312463280123, 0.21754308791766022, 0.21412025871993934, 0.2148823225976458, 0.23612262399206657, 0.2108789426301912, 0.21026770628432795, 0.22391243988349607, 0.19138519404575538, 0.18449588547834314, 0.21656259005895534, 0.1865420382591387, 0.15084031230174777, 0.17133788990721965, 0.2033092422353053, 0.17616190729361605, 0.15739202567218682, 0.148462184593244, 0.16457574961223695, 0.16044182325211676, 0.12488351734003844, 0.19419938816734636, 0.14807274399931947, 0.13369896929007719, 0.12965173082088177, 0.17741028057210406, 0.1498636236904196, 0.12351464666226494, 0.1412379748634637, 0.14322340090429542, 0.1284678075951257, 0.10693137803769706, 0.11427609030475398, 0.12962304054006993, 0.13968909943059238, 0.11700181594003932, 0.1253767950642768, 0.14608618990797295, 0.12026854973873202, 0.12009446168385506, 0.126443236850602, 0.12433139727588315, 0.10923043248317198, 0.14562940291469317, 0.1023164268874389, 0.12669685038797698, 0.13660491913867837, 0.10612489834650561, 0.09618139271682631, 0.10801576889919703, 0.10587753878673828, 0.12598426028618473, 0.11187951739567946, 0.11269720300825174, 0.12156360675501207, 0.13014083393507078, 0.0973165562219185, 0.10273016357811056, 0.10790479842060895, 0.1006800424007969, 0.12016315679260672, 0.12400889571893588, 0.1212604952100871, 0.12468117247505674, 0.11656432274397695, 0.09775594681003753, 0.09391500739237715, 0.09797476515724843, 0.11255278384568745, 0.1242137718525361, 0.08464691814910606, 0.09397608206736228, 0.13776407744636415, 0.09411659006693249, 0.10864901155265128, 0.13028449312044502, 0.11552197585549888, 0.0777974519579642, 0.08813620931202122, 0.10446573817931615, 0.09149919704068442, 0.09283756210479203, 0.0907706361694066, 0.09617017705000917, 0.07475850458683497, 0.08898132750903823, 0.09404697243611018, 0.09169675718395708, 0.10302030763318239, 0.06792158227605664, 0.12211283720341833, 0.08610031673438497, 0.073814885261133, 0.10892250068521066, 0.08950522485622792, 0.08750053869693658, 0.07551484281144179, 0.10572219453443113, 0.07326194963455475, 0.11333193312136298, 0.08307398394792341, 0.10268030187989739, 0.09861137490230579, 0.06773623502040227, 0.0816482615669527, 0.11089011413839368, 0.11264955614048175, 0.11380400022822126, 0.06785511039456633, 0.07276568037241767, 0.07424261694653103, 0.06600460400832374, 0.11362289373914565, 0.07708134619459878, 0.07432348551395429, 0.09832988033258819, 0.09117336386227186, 0.08533973659980999, 0.07789208624416215, 0.0563355840147077, 0.06830925541725165, 0.07347101197830382, 0.08364869066886368, 0.06900984795033578, 0.06703725804310309, 0.1044642862102646, 0.08770398572739643, 0.0962024210234581, 0.09997938771624347, 0.06866556526215163, 0.08406624031405364, 0.060112810479906166, 0.08652302432431863, 0.07467138910827421, 0.11284824748635999, 0.06322852063554202, 0.07242441272457972, 0.0956086115137292, 0.06690687653614849, 0.11456373550379766, 0.07756147747592543, 0.06255558777111198, 0.1013916695950829, 0.0760323854029847, 0.05864378009407251, 0.06097171085434952, 0.060631957275836035, 0.06557770468597715, 0.06802829710258801, 0.0908501699435838, 0.06121750565213886, 0.06638594286456273, 0.06814384976820223, 0.05507975978371007, 0.0635962353871682, 0.13179853123171748, 0.10091005714966197, 0.05977392251169935, 0.055266210165546115, 0.06488620501781374, 0.08782505226792356, 0.06401819378517548, 0.06924605059277496, 0.0692402318957161, 0.06643730190667904, 0.07309187459057674, 0.06438150132063808, 0.0644936917032271, 0.059743640881590156, 0.05699602777223575, 0.07469662631965403, 0.07439903162718331, 0.061935897282494955, 0.08948026094850135, 0.06387112276756682, 0.05019494193012292, 0.05198856039205522, 0.09165819136307635, 0.05555996112581952, 0.09379294364556842, 0.06765515373056663, 0.10672942237180622, 0.07511193152895347, 0.0788808157424451, 0.06437588434284898, 0.06303001753322522, 0.0701255005642488, 0.04729190179053863, 0.05476665217298177, 0.10243444691018763, 0.12444934667086209, 0.07892809286034412, 0.06268490233747019, 0.052680548691514124, 0.06513219551959032, 0.08834216012795269, 0.06633638043463644, 0.05665389732875584, 0.11088244910682674, 0.07807811725773682, 0.05336212831019164, 0.06682951097204393, 0.06280736328229045, 0.07415286934502283, 0.09453205942123356, 0.06615123580903826, 0.05366227238824235, 0.0897434932502115, 0.05203206280161644, 0.07318735268485131, 0.09356854308361875, 0.05495673168346117, 0.06988291953208034, 0.0667019530510817, 0.05386141360830867, 0.07122872549392892, 0.050254320038869545, 0.09987449355427504, 0.04544663720731557, 0.07117939991538677, 0.050118775304526304, 0.04490095972429775, 0.04501773334835932, 0.0632165212390732, 0.05876416924954459, 0.06035719604208583, 0.07033711444550665, 0.07829714544959575, 0.04805646914109865, 0.050375231722834035, 0.07039166005078411, 0.054276415699337485, 0.05005180796385601, 0.08636852118017717, 0.05370178844407662, 0.0559841911464649, 0.06453812491752844, 0.07473144388393796, 0.0961734506761711, 0.053172281164146615, 0.059863310380703055, 0.07417433152190293, 0.05055066782608115, 0.06732429057392385, 0.0952191804878361, 0.047348823591007255, 0.06765854739968646, 0.05390823437125762, 0.058215108916116774, 0.03697685027819195, 0.05882795081561543, 0.05792891279815176, 0.1074376082901388, 0.06577376984675012, 0.05251468948617676, 0.0772376977427678, 0.08591634102283523, 0.045562864644456934, 0.07763855545805792, 0.07358658313138634, 0.06488466765361595, 0.04506859764829439, 0.054900034134398465, 0.04681567602964964, 0.06270833710812962, 0.05394993693971696, 0.06326802364452036, 0.06766709939130362, 0.0625884461858173, 0.04571075979287999, 0.06475651070347485, 0.06524136487530903, 0.0616093219600853, 0.04395077627590336, 0.06727170093421701, 0.08463362294221971, 0.07270879207396899, 0.06955716783209169, 0.06096147399226564, 0.08936621654930763, 0.07873885559926741, 0.059750078902733816, 0.07769232822537983, 0.0764222987257706, 0.06775322974685533, 0.05966149907613717, 0.06596307818444427, 0.04853904441876068, 0.06360179688865976, 0.06014422918150629, 0.05300655913597837, 0.060771937426222486, 0.09492948245672746, 0.07499358111006182, 0.0681518859867393, 0.056241731653914294, 0.049310677525202865, 0.0866585332152741, 0.054284599228501505, 0.061506904977684206, 0.08373894622267444, 0.05044342866114999, 0.05502601943762984, 0.06790939941319661, 0.05890278297705062, 0.09882258695875115, 0.0813575801580616, 0.06728917352287649, 0.059640504978608865, 0.05632318642692235, 0.06452654529613379, 0.06837330434895052, 0.09189354633092157, 0.06344744959090075, 0.06771256947090326, 0.0872947598997174, 0.05868480029684388, 0.061799422119711345, 0.08618103449353723, 0.05723048080588107, 0.06000581509328523, 0.06683673561885409, 0.06477511934519227, 0.04862543214373442, 0.06604767513428084, 0.05282919477905368, 0.06912511188368986, 0.06424035452784783, 0.07971241115223837, 0.07716594131277855, 0.05548825112154038, 0.05154085854921286, 0.058873531553901425, 0.06885163339446623, 0.07467952691263784, 0.04860750296413097, 0.06775612516775215, 0.059768041791966806, 0.05939302440971101, 0.05002370762666734, 0.060831953715393364, 0.03925632332887392, 0.05349530657977912, 0.03565820051965393, 0.06668722307209993, 0.10070665757500089, 0.06887919009915885, 0.06710488908838538, 0.056286122082717924, 0.07982678027712538, 0.04610538259579915, 0.07426921101758449, 0.050664510252587856, 0.06497869660825197, 0.05847079767976311, 0.05976407975848927, 0.07024318620126314, 0.04173131428673112, 0.07510586639808642, 0.08023928483422263, 0.07997540302072917, 0.0594110733259532, 0.04409790451547611, 0.06468453044311956, 0.05255340352775307, 0.04298721603677399, 0.03991672884312137, 0.05269605109507024, 0.053372297360677085, 0.04268357601384097, 0.06624947733212466, 0.08749065813779919, 0.05068807650483908, 0.06318791646651402, 0.05111390584175261, 0.056361192635172766, 0.04846592841984342, 0.044088160869507266, 0.06816864485958181, 0.050304605537775555, 0.05906652639547177, 0.06620250959113981, 0.05783781135407816, 0.05834283170379137, 0.057483186566825134, 0.08328325397067458, 0.04953330403415372, 0.048237277836950185, 0.043801837231804225, 0.08989793554114067, 0.05490385584646373, 0.051559780702538464, 0.047832029793038786, 0.0737155297911333, 0.09193232537154965, 0.11507660473128364, 0.06835192944561388, 0.0681904492367023, 0.0871183016393798, 0.06022357280995119, 0.06213781597682224, 0.05923520306205249, 0.06185854437985488, 0.050784150853007005, 0.04492122578599283, 0.08143197871804644, 0.04952379030056646, 0.05246329358243393, 0.08745157035300856, 0.059246483180717874, 0.04444663423423345, 0.04306376421244876, 0.04343288384878469, 0.05180252184931393, 0.05799098932996466, 0.040654301568814125, 0.045245820133727806, 0.041058051984388125, 0.05146799086951215, 0.06514106355543432, 0.06071080305200935, 0.07416122986738459, 0.0862484428044915, 0.04055516102096896, 0.06152741009921033, 0.040404794729360226, 0.06309423164108628, 0.06865940413390202, 0.05659557650320559, 0.041626997427735554, 0.05159372780624241, 0.04752448067632645, 0.060291914133032594, 0.05722409829777895, 0.06537788859403972, 0.049115129909640766, 0.052710051767147485, 0.05362809733580342, 0.04633705017513486, 0.05288665792564574, 0.0566691502782135, 0.05021675965828803, 0.05517131027971669, 0.044025805908609626, 0.053819003844664294, 0.042724493324647304, 0.0653303561461163, 0.0803917540589634, 0.06143367706318902, 0.058674393324929436, 0.0493765530245785, 0.08176877814711107, 0.06292999474797525, 0.06181379560406302, 0.10026951845508064, 0.05821982763828703, 0.055840150989251984, 0.05189963179382935, 0.04808146872182167, 0.0682093863448454, 0.06886627297761264, 0.05307950110551734, 0.035257759732366396, 0.0949040580341295, 0.0909068258335117, 0.05932894296851021, 0.055276818095127654, 0.05134998521300529, 0.04744546227881123, 0.0797758505671598, 0.051043141016354085, 0.04705973456533857, 0.05211010499880702, 0.04236835878111122, 0.05850821006206071, 0.04058102796312412, 0.04816874857122338, 0.07310812303857933, 0.05472528918372006, 0.05276177409982965, 0.0615281494007832, 0.08471210389115133, 0.09848431094279625, 0.05219263380175151, 0.05160671260133631, 0.0613698655649598, 0.061332385521184106, 0.04549402736270482, 0.08799999909878535, 0.07786241217837052, 0.0633542982235202, 0.04406058697336211, 0.04997008541397534, 0.059028545280490105, 0.05868680894260451, 0.044978746160016375, 0.06336565358530569, 0.043485947899289716, 0.05986309959462111, 0.09951352569519814, 0.06066756233908542, 0.03920468964064085, 0.04858802476963746, 0.044810612784592294, 0.07423843366393033, 0.07920288451173674, 0.05304341693820641, 0.05350472610226287, 0.0748349835633772, 0.053403729984548974, 0.043798052079501056, 0.056722806261145765, 0.05418627230218258, 0.07389903486585077, 0.05109076636195421, 0.0512868938454829, 0.053524328355242845, 0.044510724288244846, 0.0461351471701754, 0.05537204303524557, 0.055742158991511595, 0.05816363351159884, 0.03856687713308811, 0.06484833708227943, 0.1174072657323204, 0.03867758441874214, 0.06212441298401643, 0.042756191269985075, 0.05979298477493367, 0.06752183491164032, 0.051942117074552714, 0.05158810881160311, 0.082911875667918]
            f1.write("\n")
            f1.write("Trained Parameters")
            f1.write("\n")
            f1.write(str(trained_params))
            f1.write("\n")
            f1.write("-------------------------------------------------------------------------------------------------------------")
            predictions = []
            iters = 3
            prcnt = 0.3
            for i in range(0,iters):
                pred = MC_QCNN_circuit.MC_QCNN(X_test, trained_params, U , U_params , Embedding, cost_fn, prcnt)
                predictions.append(pred)
                print("Iteration Done: " + str(i))

            for pred in predictions:
                accuracy = accuracy_test(pred, Y_test, cost_fn, binary)
                print("Accuracy for testing with MC Dropout :" + str(accuracy))
                f1.write("Accuracy for testing with MC Dropout" + U + " " + Encoding + " :" + str(accuracy))
                f1.write("\n")
            for pred in predictions:
                accuracy = accuracy_test(pred, Y_val, cost_fn, binary)
                print("Accuracy for validation with MC Dropout :" + str(accuracy))
                f1.write("Accuracy for validation with MC Dropout" + U + " " + Encoding + " :" + str(accuracy))
                f1.write("\n")
            f1.write("----------------------------------------------------------------------------------------------")
            '''
    f1.close()      
    f.close()

def Data_norm(dataset, classes, Encodings, binary=True):
    J = len(Encodings)
    Num_data = 10000

    f = open('Result/data_norm.txt', 'a')

    for j in range(J):
        Encoding = Encodings[j]

        X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes=classes,feature_reduction=Encoding, binary=binary)

        if Encoding == 'pca32-3' or Encoding == 'autoencoder32-3':
            norms_X1 = []
            norms_X2 = []
            for i in range(Num_data):
                index = np.random.randint(0, len(X_train))
                X = X_train[index]

                X1 = X[:2 ** 4]
                X2 = X[2 ** 4:2 ** 5]
                norm_X1, norm_X2 = np.linalg.norm(X1), np.linalg.norm(X2)
                norms_X1.append(norm_X1)
                norms_X2.append(norm_X2)

            norms_X1, norms_X2 = np.array(norms_X1), np.array(norms_X2)
            mean_X1, stdev_X1 = np.mean(norms_X1), np.std(norms_X1)
            mean_X2, stdev_X2 = np.mean(norms_X2), np.std(norms_X2)

            if Encoding == 'pca32-3':
                f.write("PCA32 Encoding\n")
            elif Encoding == 'autoencoder32-3':
                f.write("autoencoder32 Encoding\n")
            f.write("mean of X1: " + str(mean_X1) + " standard deviation of X1: " + str(stdev_X1))
            f.write("\n")
            f.write("mean of X2: " + str(mean_X2) + " standard deviation of X2: " + str(stdev_X2))
            f.write("\n")

        elif Encoding == 'pca16' or Encoding == 'autoencoder16':
            norms_X1 = []
            norms_X2 = []
            norms_X3 = []
            norms_X4 = []
            for i in range(Num_data):
                index = np.random.randint(0, len(X_train))
                X = X_train[index]

                X1 = X[:4]
                X2 = X[4:8]
                X3 = X[8:12]
                X4 = X[12:16]
                norm_X1, norm_X2, norm_X3, norm_X4 = np.linalg.norm(X1), np.linalg.norm(X2), np.linalg.norm(
                    X3), np.linalg.norm(X4)

                norms_X1.append(norm_X1)
                norms_X2.append(norm_X2)
                norms_X3.append(norm_X3)
                norms_X4.append(norm_X4)

            norms_X1, norms_X2, norms_X3, norms_X4 = np.array(norms_X1), np.array(norms_X2), np.array(norms_X3), np.array(norms_X4)

            mean_X1, stdev_X1 = np.mean(norms_X1), np.std(norms_X1)
            mean_X2, stdev_X2 = np.mean(norms_X2), np.std(norms_X2)
            mean_X3, stdev_X3 = np.mean(norms_X3), np.std(norms_X3)
            mean_X4, stdev_X4 = np.mean(norms_X4), np.std(norms_X4)

            if Encoding == 'pca16':
                f.write("PCA16 Encoding\n")
            elif Encoding == 'autoencoder16':
                f.write("autoencoder16 Encoding\n")
            f.write("mean of X1: " + str(mean_X1) + " standard deviation of X1: " + str(stdev_X1))
            f.write("\n")
            f.write("mean of X2: " + str(mean_X2) + " standard deviation of X2: " + str(stdev_X2))
            f.write("\n")
            f.write("mean of X3: " + str(mean_X3) + " standard deviation of X3: " + str(stdev_X3))
            f.write("\n")
            f.write("mean of X4: " + str(mean_X4) + " standard deviation of X4: " + str(stdev_X4))
            f.write("\n")

    f.close()